import torch
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


def main():
    args = get_args()
    set_seed(args.seed)

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
    cnn = load_pretrained_cnn(args.dqn_path, freeze=args.freeze_encoder)
    cnn = cnn.to(device)

    # Create model based on algorithm
    print(f"\nCreating {args.algo.upper()} model...")

    # Compute class weights for BC (to handle class imbalance)
    class_weights = None
    if args.algo == 'bc':
        class_weights = compute_class_weights(train_loader, action_dim=6, device=device, exponent=args.class_weight_exponent)
        model = BehaviorCloning(cnn, action_dim=6, logit_div=args.logit_div)
        train_fn = train_bc
        val_fn = val_bc

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
        # If encoder is frozen, only optimize non-encoder parameters
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr
        )
        print(f"Optimizer: Single LR={args.lr:.2e} (encoder frozen)")
    else:
        # If encoder is not frozen, use different learning rates
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

    step = 0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch"):
        # ===== TRAINING =====
        if args.algo == 'bc':
            train_loss, train_accuracy = train_fn(model, train_loader, optimizer, device, args.label_smoothing, class_weights)
            val_loss, val_accuracy = val_fn(model, val_loader, device, args.label_smoothing, class_weights)

            tqdm.write(
                f"Epoch {epoch}/{args.epochs}\n"
                f"  Train - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}\n"
                f"  Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}"
            )

            # Save model every save_interval epochs
            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), save_dir / f'epoch_{epoch}.pth')
                tqdm.write(f"  ✓ Saved model: epoch_{epoch}.pth")

        elif args.algo == 'bcq':
            train_q_loss, train_bc_loss, train_bc_accuracy, train_avg_q, step = train_fn(
                model, train_loader, optimizer, device, args.gamma, step, args.target_update_freq, args.label_smoothing
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
            train_td_loss, train_cql_loss, train_total_loss, train_avg_q, step = train_fn(
                model, train_loader, optimizer, device, args.gamma, step, args.target_update_freq
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

            eval_stats = evaluate_agent(model, args.env_name, device, args.eval_episodes, args.seed + epoch)
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
