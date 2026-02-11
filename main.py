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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_fire_move_weights(dataloader, device='cpu', exponent=0.5):
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
        # Prevent division by zero with clamp_min
        fire_counts_safe = fire_counts.float().clamp_min(1.0)
        move_counts_safe = move_counts.float().clamp_min(1.0)

        # Apply exponent to inverse frequency weights
        fire_raw_weights = total_samples / (2 * fire_counts_safe)
        fire_weights = fire_raw_weights ** exponent

        move_raw_weights = total_samples / (3 * move_counts_safe)
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
        val_file_idx=args.val_file_idx
    )
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    print("\nLoading pretrained DQN CNN (frozen)...")
    cnn = load_pretrained_cnn(args.dqn_path, freeze=True)
    cnn = cnn.to(device)

    # Create model based on algorithm
    print(f"\nCreating {args.algo.upper()} model...")

    if args.algo == 'bc':
        model = BehaviorCloning(cnn, action_dim=6)
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
        model = BCQ(cnn, action_dim=6, threshold=args.bcq_threshold, bc_path=args.bc_path)
        train_fn = train_bcq
        val_fn = val_bcq

    elif args.algo == 'cql':
        model = CQL(cnn, action_dim=6, alpha=args.cql_alpha)
        train_fn = train_cql
        val_fn = val_cql

    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    model = model.to(device)

    # Optimizer (encoder is always frozen, only optimize trainable parameters)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    print(f"Optimizer: LR={args.lr:.2e} (encoder frozen)")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*80)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch"):
        # ===== TRAINING =====
        if args.algo == 'bc':
            train_loss, train_acc, train_fire_loss, train_move_loss, train_fire_acc, train_move_acc = train_fn(
                model, train_loader, optimizer, device, args.label_smoothing, fire_weights, move_weights,
                args.fire_loss_weight, args.move_loss_weight
            )
            val_loss, val_acc, val_fire_loss, val_move_loss, val_fire_acc, val_move_acc = val_fn(
                model, val_loader, device, args.label_smoothing, fire_weights, move_weights,
                args.fire_loss_weight, args.move_loss_weight
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
            train_q_loss, train_avg_q = train_fn(
                model, train_loader, optimizer, device, args.gamma, args.target_update_freq, args.reward_scale
            )

            val_q_loss, val_avg_q = val_fn(
                model, val_loader, device, args.gamma, args.reward_scale
            )

            tqdm.write(
                f"Epoch {epoch}/{args.epochs}\n"
                f"  Train - Q Loss: {train_q_loss:.4f}, Avg Q: {train_avg_q:.2f}\n"
                f"  Val   - Q Loss: {val_q_loss:.4f}, Avg Q: {val_avg_q:.2f}"
            )

            # Save model every save_interval epochs
            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), save_dir / f'epoch_{epoch}.pth')
                tqdm.write(f"  ✓ Saved model: epoch_{epoch}.pth")

        elif args.algo == 'cql':
            train_td_loss, train_cql_loss, train_total_loss, train_avg_q = train_fn(
                model, train_loader, optimizer, device, args.gamma, args.target_update_freq, args.reward_scale
            )

            val_td_loss, val_cql_loss, val_total_loss, val_avg_q = val_cql(
                model, val_loader, device, args.gamma, args.reward_scale
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
