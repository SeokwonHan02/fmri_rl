import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

from args import get_args
from dataset import create_train_val_dataloaders
from model import (
    load_pretrained_cnn,
    BehaviorCloning, train_bc, val_bc,
    BCQ, train_bcq, val_bcq,
    CQL, train_cql, val_cql
)
from eval import evaluate_agent


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataloader, action_dim=6, device='cpu', exponent=1.0):
    """
    Compute class weights based on inverse frequency

    Args:
        dataloader: DataLoader to compute action distribution from
        action_dim: Number of action classes
        device: Device to put weights on
        exponent: Exponent to apply to weights (0.0 = no weight, 0.5 = sqrt, 1.0 = inverse frequency)

    Returns:
        torch.Tensor: Class weights for weighted cross-entropy loss
    """
    print(f"\nComputing class weights from action distribution (exponent={exponent})...")
    action_counts = torch.zeros(action_dim, dtype=torch.long)

    for batch in dataloader:
        action = batch['action']

        # Convert one-hot to index if needed
        if action.dim() == 2:
            action_idx = action.argmax(dim=-1)
        else:
            action_idx = action

        # Count each action
        for a in action_idx:
            action_counts[a] += 1

    # Compute inverse frequency weights with exponent
    total_samples = action_counts.sum().float()

    if exponent == 0.0:
        # No weighting - all weights are 1
        class_weights = torch.ones(action_dim)
    else:
        # Apply exponent to inverse frequency weights
        raw_weights = total_samples / (action_dim * action_counts.float())
        class_weights = raw_weights ** exponent

    # Print distribution
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHT+FIRE', 'LEFT+FIRE']
    print("\nAction Distribution:")
    for i in range(action_dim):
        name = action_names[i] if i < len(action_names) else f'Action {i}'
        percentage = (action_counts[i].float() / total_samples * 100).item()
        print(f"  {name:12s}: {action_counts[i]:6,} ({percentage:5.2f}%) - Weight: {class_weights[i]:.4f}")

    return class_weights.to(device)


def compute_fire_move_weights(dataloader, device='cpu', exponent=1.0):
    """
    Compute class weights for fire and move based on inverse frequency

    Args:
        dataloader: DataLoader to compute action distribution from
        device: Device to put weights on
        exponent: Exponent to apply to weights (0.0 = no weight, 0.5 = sqrt, 1.0 = inverse frequency)

    Returns:
        fire_weights: Tensor of shape (2,) for fire classes
        move_weights: Tensor of shape (3,) for move classes
    """
    from model.bc import action_to_fire_move

    print(f"\nComputing fire/move class weights from action distribution (exponent={exponent})...")
    fire_counts = torch.zeros(2, dtype=torch.long)  # fire=0, fire=1
    move_counts = torch.zeros(3, dtype=torch.long)  # move=0, move=1, move=2

    for batch in dataloader:
        action = batch['action']

        # Convert one-hot to index if needed
        if action.dim() == 2:
            action_idx = action.argmax(dim=-1)
        else:
            action_idx = action

        # Convert to fire and move labels
        fire_label, move_label = action_to_fire_move(action_idx)

        # Count each fire and move
        for f in fire_label:
            fire_counts[f] += 1
        for m in move_label:
            move_counts[m] += 1

    total_samples = fire_counts.sum().float()

    # Compute weights
    if exponent == 0.0:
        # No weighting - all weights are 1
        fire_weights = torch.ones(2)
        move_weights = torch.ones(3)
    else:
        # Apply exponent to inverse frequency weights
        fire_raw_weights = total_samples / (2 * fire_counts.float())
        fire_weights = fire_raw_weights ** exponent

        move_raw_weights = total_samples / (3 * move_counts.float())
        move_weights = move_raw_weights ** exponent

    # Print distribution
    print("\nFire Distribution:")
    fire_names = ['No Fire (NOOP, RIGHT, LEFT)', 'Fire (FIRE, RIGHT+FIRE, LEFT+FIRE)']
    for i in range(2):
        percentage = (fire_counts[i].float() / total_samples * 100).item()
        print(f"  {fire_names[i]:40s}: {fire_counts[i]:6,} ({percentage:5.2f}%) - Weight: {fire_weights[i]:.4f}")

    print("\nMove Distribution:")
    move_names = ['No Move (NOOP, FIRE)', 'Right (RIGHT, RIGHT+FIRE)', 'Left (LEFT, LEFT+FIRE)']
    for i in range(3):
        percentage = (move_counts[i].float() / total_samples * 100).item()
        print(f"  {move_names[i]:40s}: {move_counts[i]:6,} ({percentage:5.2f}%) - Weight: {move_weights[i]:.4f}")

    return fire_weights.to(device), move_weights.to(device)


def main():
    args = get_args()
    set_seed(args.seed)

    # Validate freeze options
    if args.freeze_encoder and args.freeze_conv12_only:
        raise ValueError("Cannot use both --freeze-encoder and --freeze-conv12-only. Choose one.")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir) / args.algo
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models will be saved to: {save_dir}")

    print("\nLoading data...")
    train_loader, val_loader = create_train_val_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        subject=args.subject,
        num_workers=args.num_workers,
        val_split=1.0/11  # 11개 중 1개
    )
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    print("\nLoading pretrained DQN CNN...")
    cnn = load_pretrained_cnn(
        args.dqn_path,
        freeze=args.freeze_encoder,
        freeze_conv12_only=args.freeze_conv12_only
    )
    cnn = cnn.to(device)

    # Create model based on algorithm
    print(f"\nCreating {args.algo.upper()} model...")

    if args.algo == 'bc':
        model = BehaviorCloning(cnn, action_dim=6, logit_div=args.logit_div)
        train_fn = train_bc
        val_fn = val_bc
        print("BC Loss: Fire Binary CE + Move 3-class CE")

        # Compute fire/move class weights if weight exponent > 0
        if args.class_weight_exponent > 0:
            fire_weights, move_weights = compute_fire_move_weights(
                train_loader, device, args.class_weight_exponent
            )
        else:
            fire_weights, move_weights = None, None
            print("\nNo class weighting (exponent=0.0)")

    elif args.algo == 'bcq':
        model = BCQ(cnn, action_dim=6, threshold=args.bcq_threshold, logit_div=args.logit_div, bc_path=args.bc_path)
        train_fn = train_bcq
        val_fn = val_bcq

    elif args.algo == 'cql':
        model = CQL(cnn, action_dim=6, alpha=args.cql_alpha)
        train_fn = train_cql
        val_fn = val_cql

    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    model = model.to(device)

    # Optimizer with separate learning rates for encoder and other parameters
    if args.freeze_encoder:
        # All encoder frozen, only optimize non-encoder parameters
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr
        )
        print(f"Optimizer: Single LR={args.lr:.2e} (entire encoder frozen)")
    elif args.freeze_conv12_only:
        # Conv1, Conv2 frozen, Conv3 trainable with encoder_lr
        conv3_params = list(model.cnn.cnn[4].parameters())  # Conv3
        conv3_param_ids = [id(p) for p in conv3_params]
        other_params = [p for p in model.parameters() if id(p) not in conv3_param_ids and p.requires_grad]

        optimizer = optim.Adam([
            {'params': conv3_params, 'lr': args.encoder_lr},
            {'params': other_params, 'lr': args.lr}
        ])
        print(f"Optimizer: Conv3 LR={args.encoder_lr:.2e}, Other LR={args.lr:.2e} (Conv1/Conv2 frozen)")
    else:
        # All encoder trainable with encoder_lr
        encoder_params = list(model.cnn.parameters())
        encoder_param_ids = [id(p) for p in encoder_params]
        other_params = [p for p in model.parameters() if id(p) not in encoder_param_ids and p.requires_grad]

        optimizer = optim.Adam([
            {'params': encoder_params, 'lr': args.encoder_lr},
            {'params': other_params, 'lr': args.lr}
        ])
        print(f"Optimizer: Encoder LR={args.encoder_lr:.2e}, Other LR={args.lr:.2e}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*80)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch"):
        # ===== TRAINING =====
        if args.algo == 'bc':
            train_loss, train_acc, train_fire_loss, train_move_loss, train_fire_acc, train_move_acc = train_fn(
                model, train_loader, optimizer, device, args.label_smoothing, fire_weights, move_weights
            )
            val_loss, val_acc, val_fire_loss, val_move_loss, val_fire_acc, val_move_acc = val_fn(
                model, val_loader, device, args.label_smoothing, fire_weights, move_weights
            )

            tqdm.write(
                f"Epoch {epoch}/{args.epochs}\n"
                f"  Train - Loss: {train_loss:.4f}, Action Acc: {train_acc:.4f}\n"
                f"          Fire Loss: {train_fire_loss:.4f}, Fire Acc: {train_fire_acc:.4f}\n"
                f"          Move Loss: {train_move_loss:.4f}, Move Acc: {train_move_acc:.4f}\n"
                f"  Val   - Loss: {val_loss:.4f}, Action Acc: {val_acc:.4f}\n"
                f"          Fire Loss: {val_fire_loss:.4f}, Fire Acc: {val_fire_acc:.4f}\n"
                f"          Move Loss: {val_move_loss:.4f}, Move Acc: {val_move_acc:.4f}"
            )

            # Save model every save_interval epochs
            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), save_dir / f'epoch_{epoch}.pth')
                tqdm.write(f"  ✓ Saved model: epoch_{epoch}.pth")

        elif args.algo == 'bcq':
            train_q_loss, train_bc_loss, train_bc_accuracy, train_avg_q = train_fn(
                model, train_loader, optimizer, device, args.gamma, args.target_update_freq, args.label_smoothing
            )

            val_q_loss, val_bc_loss, val_bc_accuracy, val_avg_q = val_fn(
                model, val_loader, device, args.gamma, args.label_smoothing
            )

            tqdm.write(
                f"Epoch {epoch}/{args.epochs}\n"
                f"  Train - Q Loss: {train_q_loss:.4f}, BC Loss: {train_bc_loss:.4f}, BC Acc: {train_bc_accuracy:.4f}, Avg Q: {train_avg_q:.2f}\n"
                f"  Val   - Q Loss: {val_q_loss:.4f}, BC Loss: {val_bc_loss:.4f}, BC Acc: {val_bc_accuracy:.4f}, Avg Q: {val_avg_q:.2f}"
            )

            # Save model every save_interval epochs
            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), save_dir / f'epoch_{epoch}.pth')
                tqdm.write(f"  ✓ Saved model: epoch_{epoch}.pth")

        elif args.algo == 'cql':
            train_td_loss, train_cql_loss, train_total_loss, train_avg_q = train_fn(
                model, train_loader, optimizer, device, args.gamma, args.target_update_freq
            )

            val_td_loss, val_cql_loss, val_total_loss, val_avg_q = val_cql(
                model, val_loader, device, args.gamma
            )

            tqdm.write(
                f"Epoch {epoch}/{args.epochs}\n"
                f"  Train - TD Loss: {train_td_loss:.4f}, CQL Loss: {train_cql_loss:.4f}, Total: {train_total_loss:.4f}, Avg Q: {train_avg_q:.2f}\n"
                f"  Val   - TD Loss: {val_td_loss:.4f}, CQL Loss: {val_cql_loss:.4f}, Total: {val_total_loss:.4f}, Avg Q: {val_avg_q:.2f}"
            )

            # Save model every save_interval epochs
            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), save_dir / f'epoch_{epoch}.pth')
                tqdm.write(f"  ✓ Saved model: epoch_{epoch}.pth")

        # ===== EVALUATION =====
        if epoch % args.eval_interval == 0:
            tqdm.write(f"\n{'='*80}")
            tqdm.write(f"EVALUATION at Epoch {epoch}")
            tqdm.write(f"{'='*80}")

            eval_stats = evaluate_agent(model, args.env_name, device, args.eval_episodes, args.seed + epoch, args.deterministic)
            tqdm.write(
                f"  Eval - Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}\n"
                f"         Min: {eval_stats['min_reward']:.1f}, Max: {eval_stats['max_reward']:.1f}\n"
                f"         Mean Length: {eval_stats['mean_length']:.1f}"
            )
            tqdm.write(f"{'='*80}\n")

    print("\n" + "="*80)
    print("Training complete!")
    print(f"All models saved to: {save_dir}")


if __name__ == '__main__':
    main()
