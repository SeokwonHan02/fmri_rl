"""
Measure RND-based OOD uncertainty for validation data (state-only).

For each transition in the validation set:
  uncertainty = MSE(predictor(s), target(s))   (state only, no action)

Loads the trained RND checkpoint (from train_rnd.py) and saves per-transition
uncertainty scores as a .npz file.

Usage:
    python measure_rnd_uncertainty.py \\
        --rnd_ckpt ./models/rnd/rnd_epoch30.pth \\
        --data_dir ./processed_data_frameskip_4 \\
        --subject sub_1 \\
        --output_path ./results/rnd_uncertainty_sub1_val10.npz
"""

import torch
import numpy as np
from pathlib import Path
import glob
import argparse

from dataset import OfflineRLDataset
from torch.utils.data import DataLoader

from train_rnd import RND


# ---------------------------------------------------------------------------
# Data loading (val only)
# ---------------------------------------------------------------------------

def create_val_dataloader(data_dir, batch_size, subject, num_workers=4, val_file_idx=10):
    """Load only the validation file."""
    subject_dir = Path(data_dir) / subject
    if not subject_dir.exists():
        raise ValueError(f"Subject directory not found: {subject_dir}")

    npz_files = sorted(glob.glob(str(subject_dir / '*.npz')))
    n_files = len(npz_files)

    if val_file_idx < 0 or val_file_idx >= n_files:
        raise ValueError(f"val_file_idx={val_file_idx} out of range [0, {n_files-1}]")

    val_file = npz_files[val_file_idx]
    print(f"\nValidation file: {Path(val_file).name}")

    val_dataset = OfflineRLDataset(npz_files=[val_file])

    use_pin_memory = torch.cuda.is_available()
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,    # keep original order
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    return val_loader, len(val_dataset)


# ---------------------------------------------------------------------------
# Load RND model
# ---------------------------------------------------------------------------

def load_rnd(ckpt_path, device):
    """Load RND model from checkpoint saved by train_rnd.py."""
    print(f"\nLoading RND checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    saved_args = ckpt.get('args', {})
    embed_dim  = saved_args.get('embed_dim', 512)
    obs_clip   = saved_args.get('obs_clip', 5.0)
    epoch      = ckpt.get('epoch', '?')

    print(f"  Trained for {epoch} epoch(s)")
    print(f"  embed_dim={embed_dim}, obs_clip={obs_clip}")

    model = RND(embed_dim=embed_dim, obs_clip=obs_clip)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    print(f"  RND model loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Uncertainty measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_uncertainty(model, val_loader, device, n_total):
    """
    Compute per-transition RND state uncertainty for all validation transitions.

    Returns:
        uncertainty : np.ndarray (N,) float32  — RND error per state
        action_idx  : np.ndarray (N,) int64    — action taken (for analysis)
        rewards     : np.ndarray (N,) float32  — reward
        dones       : np.ndarray (N,) float32  — episode done flag
    """
    model.eval()

    all_uncertainty = np.empty(n_total, dtype=np.float32)
    all_action_idx  = np.empty(n_total, dtype=np.int64)
    all_rewards     = np.empty(n_total, dtype=np.float32)
    all_dones       = np.empty(n_total, dtype=np.float32)

    ptr = 0
    for batch in val_loader:
        state  = batch['state'].to(device)    # (B, 4, 84, 84) uint8
        action = batch['action']              # (B, 6) float one-hot (CPU, for logging only)
        reward = batch['reward']              # (B,)
        done   = batch['done']               # (B,)

        if reward.dim() == 2:
            reward = reward.squeeze(1)
        if done.dim() == 2:
            done = done.squeeze(1)

        # State-only uncertainty (action not used by model)
        unc = model.uncertainty(state)        # (B,) float

        # Action index (for downstream analysis only)
        if action.dim() == 2:
            act_idx = action.argmax(dim=-1)
        else:
            act_idx = action.long()

        B = state.size(0)
        all_uncertainty[ptr:ptr+B] = unc.cpu().numpy()
        all_action_idx[ptr:ptr+B]  = act_idx.numpy()
        all_rewards[ptr:ptr+B]     = reward.numpy()
        all_dones[ptr:ptr+B]       = done.numpy()
        ptr += B

    assert ptr == n_total, f"Expected {n_total} transitions, got {ptr}"

    return all_uncertainty, all_action_idx, all_rewards, all_dones


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_uncertainty_stats(uncertainty, action_idx, rewards):
    """Print summary statistics of the uncertainty scores."""
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHT+FIRE', 'LEFT+FIRE']

    print(f"\n{'='*60}")
    print(f"RND State Uncertainty Statistics  (N={len(uncertainty):,})")
    print(f"{'='*60}")
    print(f"  Mean   : {uncertainty.mean():.6f}")
    print(f"  Std    : {uncertainty.std():.6f}")
    print(f"  Min    : {uncertainty.min():.6f}")
    print(f"  Max    : {uncertainty.max():.6f}")
    print(f"  Median : {np.median(uncertainty):.6f}")
    print(f"  P75    : {np.percentile(uncertainty, 75):.6f}")
    print(f"  P90    : {np.percentile(uncertainty, 90):.6f}")
    print(f"  P95    : {np.percentile(uncertainty, 95):.6f}")
    print(f"  P99    : {np.percentile(uncertainty, 99):.6f}")

    print(f"\nPer-Action Uncertainty (action taken at each state):")
    for a in range(6):
        mask = action_idx == a
        if mask.sum() == 0:
            continue
        unc_a = uncertainty[mask]
        print(f"  {action_names[a]:>12s} (n={mask.sum():5,}): "
              f"mean={unc_a.mean():.6f}, std={unc_a.std():.6f}")

    pos_mask  = rewards > 0
    zero_mask = rewards == 0
    neg_mask  = rewards < 0
    for name, mask in [('reward>0', pos_mask), ('reward=0', zero_mask), ('reward<0', neg_mask)]:
        if mask.sum() == 0:
            continue
        unc_r = uncertainty[mask]
        print(f"\nUncertainty for {name} (n={mask.sum():,}):")
        print(f"  mean={unc_r.mean():.6f}, std={unc_r.std():.6f}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Measure RND-based state OOD uncertainty on validation data'
    )

    parser.add_argument('--rnd_ckpt', type=str, required=True,
                        help='Path to trained RND checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str,
                        default='/Users/seokwon/research/fMRI_RL/processed_data_frameskip_4',
                        help='Data directory')
    parser.add_argument('--subject', type=str, default='sub_1', help='Subject ID')
    parser.add_argument('--val_file_idx', type=int, default=10,
                        help='Index of file to use for validation (0-based)')
    parser.add_argument('--output_path', type=str,
                        default='./results/rnd_uncertainty.npz',
                        help='Path to save uncertainty results (.npz)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu/mps)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    model = load_rnd(args.rnd_ckpt, device)

    print("\nLoading validation data...")
    val_loader, n_total = create_val_dataloader(
        args.data_dir, args.batch_size, args.subject,
        args.num_workers, args.val_file_idx
    )
    print(f"  Total validation transitions: {n_total:,}")
    print(f"  Val batches                 : {len(val_loader)}")

    # -----------------------------------------------------------------------
    print("\nMeasuring per-state RND uncertainty...")
    uncertainty, action_idx, rewards, dones = measure_uncertainty(
        model, val_loader, device, n_total
    )

    print_uncertainty_stats(uncertainty, action_idx, rewards)

    # -----------------------------------------------------------------------
    np.savez(
        output_path,
        uncertainty=uncertainty,   # (N,) float32 — RND state error per transition
        action_idx=action_idx,     # (N,) int64   — action taken (for analysis)
        rewards=rewards,           # (N,) float32 — reward
        dones=dones,               # (N,) float32 — episode done flag
        rnd_ckpt=str(args.rnd_ckpt),
        subject=args.subject,
        val_file_idx=args.val_file_idx,
    )
    print(f"\nSaved uncertainty results to: {output_path}")
    print(f"  Keys: uncertainty, action_idx, rewards, dones")
    print(f"  Shape: ({n_total},)")


if __name__ == '__main__':
    main()
