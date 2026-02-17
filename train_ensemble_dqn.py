"""
Train 5 Ensemble DQN models with Offline RL data
- Uses Offline RL data (FQI - Fitted Q-Iteration)
- Vanilla DQN (no conservative penalty, equivalent to CQL with alpha=0)
- Trains CNN from scratch (not frozen)
- Different random seeds for each model to create diversity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import glob
import argparse

from dataset import OfflineRLDataset
from torch.utils.data import DataLoader
from model.dqn import DQN


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_train_val_dataloaders_10train(data_dir, batch_size, subject, num_workers=4, val_file_idx=10):
    """
    Create train and validation dataloaders
    - Use 10 files for training (excluding val_file_idx)
    - Use 1 file for validation

    Args:
        data_dir: Base directory containing processed data
        batch_size: Batch size for training
        subject: Subject ID (e.g., 'sub_1')
        num_workers: Number of data loading workers
        val_file_idx: Index of file to use for validation (0-based)
    """
    # Find all npz files
    subject_dir = Path(data_dir) / subject
    if not subject_dir.exists():
        raise ValueError(f"Subject directory not found: {subject_dir}")

    npz_files = sorted(glob.glob(str(subject_dir / '*.npz')))
    n_files = len(npz_files)

    if n_files < 11:
        raise ValueError(f"Need at least 11 files, but found only {n_files} files")

    # Validate val_file_idx
    if val_file_idx < 0 or val_file_idx >= n_files:
        raise ValueError(f"val_file_idx={val_file_idx} out of range [0, {n_files-1}]")

    print(f"\nSplitting data:")
    print(f"  Total files: {n_files}")
    print(f"  Validation file index: {val_file_idx}")
    print(f"  Validation file: {Path(npz_files[val_file_idx]).name}")
    print(f"  Train files: 10 (excluding validation file)")

    # Split files
    val_files = [npz_files[val_file_idx]]
    all_train_files = npz_files[:val_file_idx] + npz_files[val_file_idx+1:]
    train_files = all_train_files[:10]  # Use only first 10 files for training

    print(f"\nTrain files used:")
    for i, f in enumerate(train_files):
        print(f"  {i+1}. {Path(f).name}")

    # Create datasets
    print(f"\nLoading training data...")
    train_dataset = OfflineRLDataset(npz_files=train_files)

    print(f"\nLoading validation data...")
    val_dataset = OfflineRLDataset(npz_files=val_files)

    # Disable pin_memory for MPS (Apple Silicon GPU)
    use_pin_memory = torch.cuda.is_available()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader


def soft_update_target(policy_net, target_net, tau):
    """Soft update of target network parameters: θ_target = τ*θ_policy + (1-τ)*θ_target"""
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)


def train_epoch(model, target_model, dataloader, optimizer, device, gamma=0.99,
                use_soft_update=True, tau=0.005, update_freq=1000, step_counter=0):
    """
    Train one epoch with vanilla DQN (offline, no penalty)

    Args:
        model: Policy network
        target_model: Target network
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device
        gamma: Discount factor
        use_soft_update: If True, use soft update (tau), else hard update (freq)
        tau: Soft update parameter (only used if use_soft_update=True)
        update_freq: Hard update frequency (only used if use_soft_update=False)
        step_counter: Current training step (for hard update)

    Returns:
        avg_loss, avg_q_value, new_step_counter
    """
    model.train()
    total_loss = 0
    total_q_value = 0
    total_samples = 0

    for batch in dataloader:
        step_counter += 1

        state = batch['state'].to(device).float() / 255.0
        action = batch['action'].to(device)
        reward = batch['reward'].to(device).float()
        next_state = batch['next_state'].to(device).float() / 255.0
        done = batch['done'].to(device).float()

        # Ensure all tensors are 1D
        if reward.dim() == 2:
            reward = reward.squeeze(1)
        if done.dim() == 2:
            done = done.squeeze(1)

        # Convert one-hot action to class index
        if action.dim() == 2:
            action_idx = action.argmax(dim=-1)
        else:
            action_idx = action

        # Forward pass
        q_values = model(state)
        q_value = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # (Batch,)

        # Compute target Q-value
        with torch.no_grad():
            next_q_values = target_model(next_state)
            next_q_value = next_q_values.max(dim=1)[0]  # (Batch,)
            target_q = reward + gamma * next_q_value * (1 - done)

        # Huber loss (smooth L1 loss) for stability
        loss = F.smooth_l1_loss(q_value, target_q)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        # Update target network
        if use_soft_update:
            soft_update_target(model, target_model, tau)
        else:
            if step_counter % update_freq == 0:
                target_model.load_state_dict(model.state_dict())

        # Statistics
        total_loss += loss.item() * state.size(0)
        total_q_value += q_values.mean().item() * state.size(0)
        total_samples += state.size(0)

    avg_loss = total_loss / total_samples
    avg_q_value = total_q_value / total_samples

    return avg_loss, avg_q_value, step_counter


def validate(model, dataloader, device, gamma=0.99):
    """Validation function"""
    model.eval()
    total_loss = 0
    total_q_value = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device).float() / 255.0
            action = batch['action'].to(device)
            reward = batch['reward'].to(device).float()
            next_state = batch['next_state'].to(device).float() / 255.0
            done = batch['done'].to(device).float()

            # Ensure all tensors are 1D
            if reward.dim() == 2:
                reward = reward.squeeze(1)
            if done.dim() == 2:
                done = done.squeeze(1)

            # Convert one-hot action to class index
            if action.dim() == 2:
                action_idx = action.argmax(dim=-1)
            else:
                action_idx = action

            # Forward pass
            q_values = model(state)
            q_value = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)

            # Compute target Q-value (use policy net as target for validation)
            next_q_values = model(next_state)
            next_q_value = next_q_values.max(dim=1)[0]
            target_q = reward + gamma * next_q_value * (1 - done)

            # Huber loss
            loss = F.smooth_l1_loss(q_value, target_q)

            # Statistics
            total_loss += loss.item() * state.size(0)
            total_q_value += q_values.mean().item() * state.size(0)
            total_samples += state.size(0)

    avg_loss = total_loss / total_samples
    avg_q_value = total_q_value / total_samples

    return avg_loss, avg_q_value


def train_single_model(seed, train_loader, val_loader, device, args, save_dir):
    """Train a single DQN model with given seed"""
    print(f"\n{'='*80}")
    print(f"Training Model with Seed {seed}")
    print(f"{'='*80}")

    # Set seed for this model
    set_seed(seed)

    # Create model
    model = DQN(action_dim=6).to(device)
    target_model = DQN(action_dim=6).to(device)
    target_model.load_state_dict(model.state_dict())

    # Freeze target model
    for param in target_model.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Model initialized with seed {seed}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Optimizer: Adam, LR={args.lr:.2e}")
    print(f"  Target update: {'Soft (tau=' + str(args.tau) + ')' if args.use_soft_update else 'Hard (freq=' + str(args.target_update_freq) + ')'}")

    # Training loop
    step_counter = 0
    for epoch in tqdm(range(1, args.epochs + 1), desc=f"Seed {seed}", unit="epoch"):
        # Train
        train_loss, train_q, step_counter = train_epoch(
            model, target_model, train_loader, optimizer, device,
            gamma=args.gamma,
            use_soft_update=args.use_soft_update,
            tau=args.tau,
            update_freq=args.target_update_freq,
            step_counter=step_counter
        )

        # Validate
        val_loss, val_q = validate(model, val_loader, device, gamma=args.gamma)

        # Log every epochs
        tqdm.write(
            f"  Epoch {epoch}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Q: {train_q:.2f} | "
            f"Val Loss: {val_loss:.4f}, Q: {val_q:.2f}"
        )

        # Save checkpoints
        is_save_epoch = (epoch % args.save_interval == 0) or (epoch == args.epochs)
        if is_save_epoch:
            save_path = save_dir / f'model_seed{seed}_epoch{epoch}.pth'
            tmp_path = save_path.with_suffix('.tmp')
            # Sync device before saving to ensure all async ops are done
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            # Move state_dict to CPU before saving (avoids MPS serialization issues)
            cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state_dict, tmp_path)
            tmp_path.replace(save_path)  # atomic rename: avoids corrupted files on crash

    print(f"✓ Model seed {seed} training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train Ensemble DQN for Epistemic Uncertainty')

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='/Users/seokwon/research/fMRI_RL/processed_data_frameskip_4',
                        help='Data directory')
    parser.add_argument('--subject', type=str, default='sub_1', help='Subject ID')
    parser.add_argument('--val_file_idx', type=int, default=10,
                        help='Index of file to use for validation')

    # Model
    parser.add_argument('--ensemble_size', type=int, default=5, help='Number of ensemble models')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46],
                        help='Random seeds for ensemble models')

    # Training
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')

    # Target network update (Hard update)
    parser.add_argument('--use_soft_update', action='store_true', default=False,
                        help='Use soft update instead of hard update')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update parameter (only used if use_soft_update=True)')
    parser.add_argument('--target_update_freq', type=int, default=1000,
                        help='Hard update frequency (only used if use_soft_update=False)')

    # System
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu/mps)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--save_dir', type=str, default='./models/ensemble_dqn',
                        help='Directory to save models')

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models will be saved to: {save_dir}")

    # Load data
    print("\nLoading data...")
    train_loader, val_loader = create_train_val_dataloaders_10train(
        args.data_dir,
        args.batch_size,
        args.subject,
        args.num_workers,
        args.val_file_idx
    )
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Ensure we have correct number of seeds
    if len(args.seeds) != args.ensemble_size:
        print(f"Warning: Number of seeds ({len(args.seeds)}) != ensemble_size ({args.ensemble_size})")
        args.seeds = args.seeds[:args.ensemble_size]

    print(f"\n{'='*80}")
    print(f"Training {args.ensemble_size} Ensemble Models")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*80}")

    # Train ensemble models
    for seed in args.seeds:
        train_single_model(seed, train_loader, val_loader, device, args, save_dir)

    print(f"\n{'='*80}")
    print("All 5 Ensemble Models Training Complete!")
    print(f"Models saved to: {save_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
