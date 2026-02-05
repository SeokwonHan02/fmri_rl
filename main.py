import torch
import torch.optim as optim
import numpy as np
import random
from pathlib import Path
import os
from tqdm import tqdm

from args import get_args
from dataset import create_dataloader
from model import (
    load_pretrained_cnn,
    BehaviorCloning, train_bc,
    BCQ, train_bcq,
    CQL, train_cql
)


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
    print(f"Checkpoints will be saved to: {save_dir}")

    print("\nLoading data...")
    dataloader = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        subject=args.subject,
        num_workers=args.num_workers,
        shuffle=True
    )
    print(f"Total batches per epoch: {len(dataloader)}")

    print("\nLoading pretrained DQN CNN...")
    cnn = load_pretrained_cnn(args.dqn_path)
    cnn = cnn.to(device)

    # Create model based on algorithm
    print(f"\nCreating {args.algo.upper()} model...")

    if args.algo == 'bc':
        model = BehaviorCloning(cnn, hidden_dim=512, action_dim=6)
        train_fn = train_bc

    elif args.algo == 'bcq':
        model = BCQ(cnn, hidden_dim=512, action_dim=6, threshold=args.bcq_threshold)
        train_fn = train_bcq

    elif args.algo == 'cql':
        model = CQL(cnn, hidden_dim=512, action_dim=6, alpha=args.cql_alpha)
        train_fn = train_cql

    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # Learning rate scheduler (Cosine Annealing)
    total_steps = args.epochs * len(dataloader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * args.lr_decay_factor
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*80)

    best_metric = float('inf') if args.algo == 'bc' else float('-inf')
    step = 0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch"):
        current_lr = optimizer.param_groups[0]['lr']

        # Train
        if args.algo == 'bc':
            loss, accuracy = train_fn(model, dataloader, optimizer, device, scheduler)
            tqdm.write(f"Epoch {epoch}/{args.epochs} | LR: {current_lr:.2e} | Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

            # Save best model based on accuracy
            if accuracy > best_metric:
                best_metric = accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                    'loss': loss,
                }, save_dir / 'best_model.pth')
                tqdm.write(f"  ✓ Saved best model (accuracy: {accuracy:.4f})")

        elif args.algo == 'bcq':
            q_loss, bc_loss, bc_accuracy = train_fn(model, dataloader, optimizer, device, args.gamma, scheduler)
            tqdm.write(f"Epoch {epoch}/{args.epochs} | LR: {current_lr:.2e} | Q Loss: {q_loss:.4f}, BC Loss: {bc_loss:.4f}, BC Accuracy: {bc_accuracy:.4f}")

            # Update target network
            if step % args.target_update_freq == 0:
                model.update_target()
                tqdm.write(f"  ✓ Updated target network")

            # Save best model based on BC accuracy
            if bc_accuracy > best_metric:
                best_metric = bc_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'q_loss': q_loss,
                    'bc_loss': bc_loss,
                    'bc_accuracy': bc_accuracy,
                }, save_dir / 'best_model.pth')
                tqdm.write(f"  ✓ Saved best model (BC accuracy: {bc_accuracy:.4f})")

        elif args.algo == 'cql':
            td_loss, cql_loss, total_loss = train_fn(model, dataloader, optimizer, device, args.gamma, scheduler)
            tqdm.write(f"Epoch {epoch}/{args.epochs} | LR: {current_lr:.2e} | TD Loss: {td_loss:.4f}, CQL Loss: {cql_loss:.4f}, Total Loss: {total_loss:.4f}")

            # Update target network
            if step % args.target_update_freq == 0:
                model.update_target()
                tqdm.write(f"  ✓ Updated target network")

            # Save best model based on lowest total loss
            if total_loss < best_metric or best_metric == float('-inf'):
                best_metric = total_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'td_loss': td_loss,
                    'cql_loss': cql_loss,
                    'total_loss': total_loss,
                }, save_dir / 'best_model.pth')
                tqdm.write(f"  ✓ Saved best model (total loss: {total_loss:.4f})")

        step += len(dataloader)

        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir / f'checkpoint_epoch_{epoch}.pth')
            tqdm.write(f"  ✓ Saved checkpoint")

    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best model saved to: {save_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
