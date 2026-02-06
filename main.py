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
    cnn = load_pretrained_cnn(args.dqn_path)
    cnn = cnn.to(device)

    # Create model based on algorithm
    print(f"\nCreating {args.algo.upper()} model...")

    if args.algo == 'bc':
        model = BehaviorCloning(cnn, hidden_dim=512, action_dim=6)
        train_fn = train_bc
        val_fn = val_bc

    elif args.algo == 'bcq':
        model = BCQ(cnn, hidden_dim=512, action_dim=6, threshold=args.bcq_threshold)
        train_fn = train_bcq
        val_fn = val_bcq

    elif args.algo == 'cql':
        model = CQL(cnn, hidden_dim=512, action_dim=6, alpha=args.cql_alpha)
        train_fn = train_cql
        val_fn = val_cql

    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # Learning rate scheduler (Cosine Annealing)
    total_steps = args.epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * args.lr_decay_factor
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*80)

    step = 0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch"):
        current_lr = optimizer.param_groups[0]['lr']

        # ===== TRAINING =====
        if args.algo == 'bc':
            train_loss, train_accuracy = train_fn(model, train_loader, optimizer, device, scheduler)
            val_loss, val_accuracy = val_fn(model, val_loader, device)

            tqdm.write(
                f"Epoch {epoch}/{args.epochs} | LR: {current_lr:.2e}\n"
                f"  Train - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}\n"
                f"  Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}"
            )

            # Save model for this epoch
            torch.save(model.state_dict(), save_dir / f'epoch_{epoch}.pth')

        elif args.algo == 'bcq':
            train_q_loss, train_bc_loss, train_bc_accuracy, train_avg_q, step = train_fn(
                model, train_loader, optimizer, device, args.gamma, scheduler, step, args.target_update_freq
            )

            val_q_loss, val_bc_loss, val_bc_accuracy, val_avg_q = val_fn(
                model, val_loader, device, args.gamma
            )

            tqdm.write(
                f"Epoch {epoch}/{args.epochs} | LR: {current_lr:.2e}\n"
                f"  Train - Q Loss: {train_q_loss:.4f}, BC Loss: {train_bc_loss:.4f}, BC Acc: {train_bc_accuracy:.4f}, Avg Q: {train_avg_q:.2f}\n"
                f"  Val   - Q Loss: {val_q_loss:.4f}, BC Loss: {val_bc_loss:.4f}, BC Acc: {val_bc_accuracy:.4f}, Avg Q: {val_avg_q:.2f}"
            )

            # Save model for this epoch
            torch.save(model.state_dict(), save_dir / f'epoch_{epoch}.pth')

        elif args.algo == 'cql':
            train_td_loss, train_cql_loss, train_total_loss, train_avg_q, step = train_fn(
                model, train_loader, optimizer, device, args.gamma, scheduler, step, args.target_update_freq
            )

            val_td_loss, val_cql_loss, val_total_loss, val_avg_q = val_cql(
                model, val_loader, device, args.gamma
            )

            tqdm.write(
                f"Epoch {epoch}/{args.epochs} | LR: {current_lr:.2e}\n"
                f"  Train - TD Loss: {train_td_loss:.4f}, CQL Loss: {train_cql_loss:.4f}, Total: {train_total_loss:.4f}, Avg Q: {train_avg_q:.2f}\n"
                f"  Val   - TD Loss: {val_td_loss:.4f}, CQL Loss: {val_cql_loss:.4f}, Total: {val_total_loss:.4f}, Avg Q: {val_avg_q:.2f}"
            )

            # Save model for this epoch
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
