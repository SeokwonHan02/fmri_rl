"""
Train Bootstrapped Ensemble DQN models with Offline RL data.

Bootstrapped DQN:
  - Split train files into `ensemble_size` partitions.
  - Model i trains on all partitions except partition i (leave-one-out).
  - No two models share the same training subset.
  - Uncertainty is measured as variance across Q-value predictions.
"""

import torch
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


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data split helpers
# ---------------------------------------------------------------------------

def partition_files(files, n_partitions):
    """Round-robin assignment of files to n_partitions partitions."""
    partitions = [[] for _ in range(n_partitions)]
    for i, f in enumerate(files):
        partitions[i % n_partitions].append(f)
    return partitions


def prepare_data_split(data_dir, subject, train_indices, test_file_idx, ensemble_size):
    """
    Split data into bootstrap partitions for the ensemble.

    Args:
        train_indices : list[int] – file indices to use for training
        test_file_idx : int – file index to use for test

    Returns:
        partitions  : list[list[str]] – ensemble_size partitions of train files
        test_files  : list[str]
    """
    subject_dir = Path(data_dir) / subject
    if not subject_dir.exists():
        raise ValueError(f"Subject directory not found: {subject_dir}")

    npz_files = sorted(glob.glob(str(subject_dir / '*.npz')))
    n_files = len(npz_files)

    if n_files == 0:
        raise ValueError(f"No npz files found in {subject_dir}")

    for idx in train_indices:
        if idx < 0 or idx >= n_files:
            raise ValueError(f"train_idx {idx} out of range [0, {n_files-1}]")
    if test_file_idx < 0 or test_file_idx >= n_files:
        raise ValueError(f"test_file_idx={test_file_idx} out of range [0, {n_files-1}]")
    if test_file_idx in train_indices:
        raise ValueError(f"test_file_idx={test_file_idx} must not overlap with train_idx")

    train_files = [npz_files[i] for i in sorted(train_indices)]
    test_files  = [npz_files[test_file_idx]]

    n_train = len(train_files)
    if n_train < ensemble_size:
        print(
            f"[Warning] Only {n_train} train files but ensemble_size={ensemble_size}. "
            f"Some models may share identical training data."
        )

    partitions = partition_files(train_files, ensemble_size)

    print(f"\nBootstrap data split:")
    print(f"  Total files      : {n_files}")
    print(f"  Train indices    : {sorted(train_indices)}  ({n_train} files)")
    print(f"  Test  file       : {Path(test_files[0]).name}  (index {test_file_idx})")
    print(f"  Ensemble size    : {ensemble_size}")
    print(f"  Partitions (each model excludes one):")
    for i, p in enumerate(partitions):
        names = [Path(f).name for f in p]
        print(f"    partition {i:2d}: {names}")

    return partitions, test_files


def make_loader(files, batch_size, num_workers, shuffle, verbose_label=None):
    if verbose_label:
        print(f"\nLoading {verbose_label} ({len(files)} files)...")
    dataset = OfflineRLDataset(npz_files=files)
    use_pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def soft_update_target(policy_net, target_net, tau):
    for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
        tp.data.copy_(tau * pp.data + (1.0 - tau) * tp.data)


def train_epoch(model, target_model, dataloader, optimizer, device,
                gamma=0.99, use_soft_update=True, tau=0.005,
                update_freq=1000, step_counter=0):
    model.train()
    total_loss = total_q = total_samples = 0

    for batch in dataloader:
        step_counter += 1

        state      = batch['state'].to(device).float() / 255.0
        action     = batch['action'].to(device)
        reward     = batch['reward'].to(device).float()
        next_state = batch['next_state'].to(device).float() / 255.0
        done       = batch['done'].to(device).float()

        if reward.dim() == 2: reward = reward.squeeze(1)
        if done.dim()   == 2: done   = done.squeeze(1)

        action_idx = action.argmax(dim=-1) if action.dim() == 2 else action

        q_values = model(state)
        q_value  = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_value = target_model(next_state).max(dim=1)[0]
            target_q     = reward + gamma * next_q_value * (1 - done)

        loss = F.smooth_l1_loss(q_value, target_q)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if use_soft_update:
            soft_update_target(model, target_model, tau)
        elif step_counter % update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        bs = state.size(0)
        total_loss    += loss.item() * bs
        total_q       += q_values.mean().item() * bs
        total_samples += bs

    return total_loss / total_samples, total_q / total_samples, step_counter


def evaluate(model, dataloader, device):
    """
    Returns test metrics:
      avg_ce, action_accuracy
    """
    model.eval()
    total_ce = total_correct = total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            state  = batch['state'].to(device).float() / 255.0
            action = batch['action'].to(device)

            action_idx = action.argmax(dim=-1) if action.dim() == 2 else action

            q_values = model(state)

            # Z-score normalize Q-values → use as logits for CE
            q_mean = q_values.mean(dim=-1, keepdim=True)
            q_std  = q_values.std(dim=-1, keepdim=True) + 1e-8
            z_vals = (q_values - q_mean) / q_std

            per_sample_ce = F.cross_entropy(z_vals, action_idx, reduction='none')

            bs = state.size(0)
            total_ce      += per_sample_ce.sum().item()
            total_correct += (z_vals.argmax(dim=-1) == action_idx).sum().item()
            total_samples += bs

    return total_ce / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# Single model training
# ---------------------------------------------------------------------------

def train_single_model(model_idx, excluded_partition, train_files, test_loader,
                       device, args, save_dir):
    """
    Train one ensemble member.

    Args:
        model_idx          : int, index of this ensemble member (0-based)
        excluded_partition : list[str], files NOT used for this model (for logging)
        train_files        : list[str], files used for training this model
        test_loader        : DataLoader for test
        device             : torch.device
        args               : parsed arguments
        save_dir           : Path to save checkpoints
    """
    seed = args.base_seed + model_idx

    print(f"\n{'='*80}")
    print(f"Model {model_idx:2d} / {args.ensemble_size - 1}  |  seed={seed}")
    print(f"  Training on  : {len(train_files)} files")
    print(f"  Excluded     : {[Path(f).name for f in excluded_partition]}")
    print(f"{'='*80}")

    set_seed(seed)

    train_loader = make_loader(
        train_files, args.batch_size, args.num_workers, shuffle=True,
        verbose_label=f"model {model_idx} train data"
    )

    model        = DQN(action_dim=6).to(device)
    target_model = DQN(action_dim=6).to(device)
    target_model.load_state_dict(model.state_dict())
    for p in target_model.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}  |  "
          f"LR={args.lr:.2e}  |  "
          f"Target: {'soft τ=' + str(args.tau) if args.use_soft_update else 'hard freq=' + str(args.target_update_freq)}")

    step_counter = 0
    summary_rows = []

    for epoch in tqdm(range(1, args.epochs + 1), desc=f"Model {model_idx}", unit="epoch"):
        train_loss, train_q, step_counter = train_epoch(
            model, target_model, train_loader, optimizer, device,
            gamma=args.gamma,
            use_soft_update=args.use_soft_update,
            tau=args.tau,
            update_freq=args.target_update_freq,
            step_counter=step_counter,
        )

        test_ce, test_acc = evaluate(model, test_loader, device)

        summary_rows.append({'epoch': epoch, 'test_ce': test_ce, 'test_acc': test_acc})

        tqdm.write(
            f"  [Model {model_idx:2d}] Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss={train_loss:.4f} Q={train_q:.2f} | "
            f"Test CE={test_ce:.4f} Acc={test_acc:.4f}"
        )

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            save_path = save_dir / f'model_{model_idx:02d}_epoch{epoch}.pth'
            tmp_path  = save_path.with_suffix('.tmp')
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_sd, tmp_path)
            tmp_path.replace(save_path)

    last = summary_rows[-1]
    print(f"\n  Model {model_idx:2d} done | "
          f"Final Test CE={last['test_ce']:.4f} Acc={last['test_acc']:.4f}")

    return summary_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Bootstrapped Ensemble DQN')

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='/Users/seokwon/research/fMRI_RL/processed_data_frameskip_4')
    parser.add_argument('--subject', type=str, default='sub_1')
    parser.add_argument('--train_idx', type=int, nargs='+', default=[0, 1, 2, 3, 4, 7, 8, 9, 10, 11],
                        help='File indices to use for training (space-separated, e.g. --train_idx 0 1 2 3 4)')
    parser.add_argument('--test_file_idx', type=int, default=6,
                        help='Index of test file (0-based)')

    # Ensemble
    parser.add_argument('--ensemble_size', type=int, default=20,
                        help='Number of ensemble members (= number of bootstrap partitions)')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base random seed; model i uses base_seed + i')

    # Training
    parser.add_argument('--epochs',     type=int,   default=25)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--gamma',      type=float, default=0.99)

    # Target network
    parser.add_argument('--use_soft_update',    action='store_true', default=False)
    parser.add_argument('--tau',                type=float, default=0.005)
    parser.add_argument('--target_update_freq', type=int,   default=1000)

    # System
    parser.add_argument('--device',        type=str, default='cuda')
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs (last epoch always saved)')
    parser.add_argument('--save_dir', type=str, default='./models/ensemble_dqn')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints → {save_dir}")

    partitions, test_files = prepare_data_split(
        args.data_dir, args.subject,
        args.train_idx, args.test_file_idx,
        args.ensemble_size,
    )

    test_loader = make_loader(test_files, args.batch_size, args.num_workers,
                              shuffle=False, verbose_label="test")

    print(f"\n{'='*80}")
    print(f"Training {args.ensemble_size} bootstrapped ensemble models")
    print(f"Each model trains on {args.ensemble_size - 1}/{args.ensemble_size} partitions (leave-one-out)")
    print(f"{'='*80}")

    all_summaries = {}

    for model_idx in range(args.ensemble_size):
        excluded = partitions[model_idx]
        model_train_files = [f for j, p in enumerate(partitions) if j != model_idx for f in p]

        summary = train_single_model(
            model_idx=model_idx,
            excluded_partition=excluded,
            train_files=model_train_files,
            test_loader=test_loader,
            device=device,
            args=args,
            save_dir=save_dir,
        )
        all_summaries[model_idx] = summary

    # Final summary table
    print(f"\n{'='*80}")
    print("ENSEMBLE TRAINING COMPLETE – Final epoch metrics per model")
    print(f"{'='*80}")
    print(f"{'Model':>6}  {'Test CE':>8}  {'Test Acc':>9}")
    print("-" * 30)
    for idx, rows in all_summaries.items():
        last = rows[-1]
        print(f"  {idx:4d}  {last['test_ce']:8.4f}  {last['test_acc']:9.4f}")

    test_accs = [r[-1]['test_acc'] for r in all_summaries.values()]
    test_ces  = [r[-1]['test_ce']  for r in all_summaries.values()]
    print("-" * 30)
    print(f"  {'Mean':>4}  {np.mean(test_ces):8.4f}  {np.mean(test_accs):9.4f}")
    print(f"  {'Std':>4}  {np.std(test_ces):8.4f}  {np.std(test_accs):9.4f}")
    print(f"\nAll checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
