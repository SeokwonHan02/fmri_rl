"""
Train Random Network Distillation (RND) for state-based OOD uncertainty quantification.

Pipeline per state s:
  1. state (B,4,84,84) uint8 -> take last frame -> (B,1,84,84)
     -> /255 -> pixel-wise normalization (pre-computed mean/std) + clip[-5,5]
  2. TARGET  (frozen):   Nature DQN CNN(1ch) -> Linear(3136,512) [no ReLU] -> f(s)
  3. PREDICTOR (trained): same architecture, separate weights   -> f_hat(s)
  4. RND error = MSE(f_hat(s), f(s))   per sample   (high = OOD state)

Frozen:   target  (random init, never updated)
Trainable: predictor (same arch, learns to replicate target on train states)

Usage:
    python train_rnd.py --data_dir ./processed_data_frameskip_4 --subject sub_1
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
# Data loading 
# ---------------------------------------------------------------------------

def create_train_val_dataloaders(data_dir, batch_size, subject, num_workers=4, val_file_idx=10):
    """
    Use 10 train files + 1 val file (leave-one-out style).
    Mirrors create_train_val_dataloaders_10train in train_ensemble_dqn.py.
    """
    subject_dir = Path(data_dir) / subject
    if not subject_dir.exists():
        raise ValueError(f"Subject directory not found: {subject_dir}")

    npz_files = sorted(glob.glob(str(subject_dir / '*.npz')))
    n_files = len(npz_files)

    if n_files < 11:
        raise ValueError(f"Need at least 11 files, found {n_files}")
    if val_file_idx < 0 or val_file_idx >= n_files:
        raise ValueError(f"val_file_idx={val_file_idx} out of range [0, {n_files-1}]")

    print(f"\nSplitting data:")
    print(f"  Total files    : {n_files}")
    print(f"  Val file index : {val_file_idx}")
    print(f"  Val file       : {Path(npz_files[val_file_idx]).name}")
    print(f"  Train files    : 10 (excluding validation file)")

    val_files = [npz_files[val_file_idx]]
    all_train_files = npz_files[:val_file_idx] + npz_files[val_file_idx + 1:]
    train_files = all_train_files[:10]

    print(f"\nTrain files:")
    for i, f in enumerate(train_files):
        print(f"  {i+1}. {Path(f).name}")

    print(f"\nLoading training data...")
    train_dataset = OfflineRLDataset(npz_files=train_files)

    print(f"\nLoading validation data...")
    val_dataset = OfflineRLDataset(npz_files=val_files)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# RND Model Components
# ---------------------------------------------------------------------------

class ObsNormalizer(nn.Module):
    """
    Per-pixel running mean/std normalization for single-frame (1, 84, 84) states.

    Statistics stored as buffers → saved/loaded with model state_dict.
    Pre-computed once from the full training set before training begins.

    normalize(x): (x - mean) / std, clipped to [-clip, clip]
    """
    def __init__(self, shape=(1, 84, 84), clip=5.0, epsilon=1e-8):
        super().__init__()
        self.clip    = clip
        self.epsilon = epsilon

        self.register_buffer('mean',  torch.zeros(shape, dtype=torch.float64))
        self.register_buffer('var',   torch.ones(shape,  dtype=torch.float64))
        self.register_buffer('count', torch.tensor(0,    dtype=torch.float64))

    @torch.no_grad()
    def update(self, x_float):
        """
        Update running statistics (Chan's parallel algorithm).
        x_float: (B, 1, 84, 84) float32 in [0, 1]
        """
        x = x_float.double().cpu()
        B = x.shape[0]

        batch_mean = x.mean(dim=0)
        batch_var  = x.var(dim=0, unbiased=False)

        mean  = self.mean.cpu()
        var   = self.var.cpu()
        count = self.count.cpu()

        total = count + B
        delta = batch_mean - mean

        new_mean = mean + delta * (B / total)

        m_a = var   * count
        m_b = batch_var * B
        m2  = m_a + m_b + delta.pow(2) * (count * B / total)

        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def normalize(self, x_float):
        """
        x_float: (B, 1, 84, 84) float32 in [0, 1]
        Returns: (B, 1, 84, 84) float32, normalized and clipped
        """
        mean = self.mean.float().to(x_float.device)
        std  = (self.var.float() + self.epsilon).sqrt().to(x_float.device)
        return ((x_float - mean) / std).clamp(-self.clip, self.clip)


class RNDNetwork(nn.Module):
    """
    Nature DQN architecture for a single input frame: (B,1,84,84) -> (B,512)

      Conv2d(1,32,8,4) -> ReLU
      Conv2d(32,64,4,2) -> ReLU
      Conv2d(64,64,3,1) -> ReLU
      Flatten -> 3136
      Linear(3136, 512)   [no ReLU: signed output preserves full random feature space]

    freeze=True  -> target network  (fixed random embedding)
    freeze=False -> predictor network (fully trainable)
    """
    def __init__(self, embed_dim=512, freeze=False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),   # (1,84,84) -> (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64,7,7)
            nn.ReLU(),
            nn.Flatten()                                  # -> 3136
        )
        self.fc = nn.Linear(3136, embed_dim)              # no ReLU after

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x_norm):
        """x_norm: (B,1,84,84) float, normalized+clipped"""
        return self.fc(self.cnn(x_norm))   # (B, embed_dim)


class RND(nn.Module):
    """
    Full RND model for state-based uncertainty.

    target    (frozen):   RNDNetwork(freeze=True)  -> f(s)
    predictor (trained):  RNDNetwork(freeze=False) -> f_hat(s)
    uncertainty = MSE(f_hat(s), f(s))  per sample
    """
    def __init__(self, embed_dim=512, obs_clip=5.0):
        super().__init__()
        self.obs_normalizer = ObsNormalizer(shape=(1, 84, 84), clip=obs_clip)
        self.target         = RNDNetwork(embed_dim=embed_dim, freeze=True)
        self.predictor      = RNDNetwork(embed_dim=embed_dim, freeze=False)

    def _preprocess(self, state):
        """
        state: (B,4,84,84) uint8
        Returns: (B,1,84,84) float32, normalized+clipped
        """
        x = state[:, -1:, :, :].float() / 255.0   # last (most recent) frame
        return self.obs_normalizer.normalize(x)

    def update_obs_stats(self, state):
        """Update running normalizer stats. Used by precompute_obs_stats() before training."""
        x = state[:, -1:, :, :].float() / 255.0
        self.obs_normalizer.update(x)

    def compute_loss(self, state):
        """
        Returns (scalar mean loss, per-sample loss (B,)).
        Gradients flow through predictor only.
        """
        x_norm = self._preprocess(state)

        with torch.no_grad():
            target_feat = self.target(x_norm)      # (B, embed_dim) — frozen

        pred_feat = self.predictor(x_norm)         # (B, embed_dim) — trainable

        per_sample = F.mse_loss(pred_feat, target_feat, reduction='none').mean(dim=-1)  # (B,)
        return per_sample.mean(), per_sample

    @torch.no_grad()
    def uncertainty(self, state):
        """Per-sample RND error (no grad). Higher = more OOD."""
        x_norm      = self._preprocess(state)
        target_feat = self.target(x_norm)
        pred_feat   = self.predictor(x_norm)
        return F.mse_loss(pred_feat, target_feat, reduction='none').mean(dim=-1)  # (B,)


# ---------------------------------------------------------------------------
# Pre-compute obs stats + Train / Validate
# ---------------------------------------------------------------------------

def precompute_obs_stats(model, train_loader):
    """
    Compute observation normalization statistics from the entire training dataset
    (one pass, CPU-only). Must be called once before training begins.
    """
    print("\nPre-computing observation normalization stats from full training set...")
    model.obs_normalizer.mean.zero_()
    model.obs_normalizer.var.fill_(1.0)
    model.obs_normalizer.count.zero_()

    for batch in tqdm(train_loader, desc="  Scanning train data", unit="batch", leave=False):
        state = batch['state']   # (B, 4, 84, 84) uint8, CPU
        model.update_obs_stats(state)

    n        = model.obs_normalizer.count.item()
    mean_val = model.obs_normalizer.mean.mean().item()
    std_val  = (model.obs_normalizer.var + model.obs_normalizer.epsilon).sqrt().mean().item()
    print(f"  Done. Samples seen: {int(n):,}")
    print(f"  Mean pixel (last frame): {mean_val:.4f}")
    print(f"  Std  pixel (last frame): {std_val:.4f}")


def train_epoch(model, dataloader, optimizer, device):
    """
    Train RND predictor for one epoch.
    Obs normalization stats are pre-computed (not updated here).
    Only predictor weights are updated.
    """
    model.train()
    model.target.eval()   # frozen target: always eval

    total_loss    = 0.0
    total_samples = 0

    for batch in dataloader:
        state = batch['state'].to(device)    # (B, 4, 84, 84) uint8

        loss, _ = model.compute_loss(state)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 10.0)
        optimizer.step()

        total_loss    += loss.item() * state.size(0)
        total_samples += state.size(0)

    return total_loss / total_samples


def validate(model, dataloader, device):
    """Compute average RND error on validation data (no stat update)."""
    model.eval()

    total_loss    = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device)

            _, per_sample = model.compute_loss(state)

            total_loss    += per_sample.sum().item()
            total_samples += state.size(0)

    return total_loss / total_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train RND for state OOD uncertainty')

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='/Users/seokwon/research/fMRI_RL/processed_data_frameskip_4',
                        help='Data directory')
    parser.add_argument('--subject', type=str, default='sub_1', help='Subject ID')
    parser.add_argument('--val_file_idx', type=int, default=10,
                        help='Index of file to use for validation (0-based)')

    # Model
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Output embedding dim (target and predictor)')
    parser.add_argument('--obs_clip', type=float, default=5.0,
                        help='Clip range for normalized observations [-clip, clip]')

    # Training
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for predictor')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # System
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu/mps)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--save_dir', type=str, default='./models/rnd',
                        help='Directory to save RND checkpoints')

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    set_seed(args.seed)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")

    # -----------------------------------------------------------------------
    # Data
    print("\nLoading data...")
    train_loader, val_loader = create_train_val_dataloaders(
        args.data_dir, args.batch_size, args.subject,
        args.num_workers, args.val_file_idx
    )
    print(f"Train batches per epoch : {len(train_loader)}")
    print(f"Val batches             : {len(val_loader)}")

    # -----------------------------------------------------------------------
    # Model
    print("\nBuilding RND model...")
    model = RND(embed_dim=args.embed_dim, obs_clip=args.obs_clip).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.predictor.parameters())

    print(f"\n  Architecture (Nature DQN, 1-frame input):")
    print(f"    State input    : last frame of 4-stack  (B,4,84,84) -> (B,1,84,84)")
    print(f"    ObsNormalizer  : pre-computed per-pixel mean/std, clip={args.obs_clip}")
    print(f"    target         : Conv(1ch)x3 -> ReLU -> Flatten -> Linear(3136,{args.embed_dim})  [frozen]")
    print(f"    predictor      : Conv(1ch)x3 -> ReLU -> Flatten -> Linear(3136,{args.embed_dim})  [trained]")
    print(f"    uncertainty    : MSE(predictor(s), target(s))  per state")
    print(f"\n  Parameters:")
    print(f"    Total          : {total_params:,}")
    print(f"    Trainable      : {trainable_params:,}  (predictor only)")

    optimizer = optim.Adam(model.predictor.parameters(), lr=args.lr)
    print(f"    Optimizer      : Adam, LR={args.lr:.2e}")

    # -----------------------------------------------------------------------
    # Pre-compute obs normalization stats from full training dataset (one pass)
    precompute_obs_stats(model, train_loader)

    # -----------------------------------------------------------------------
    # Training loop
    print(f"\nStarting RND training for {args.epochs} epochs...")
    print("=" * 80)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training RND", unit="epoch"):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = validate(model, val_loader, device)

        tqdm.write(
            f"  Epoch {epoch:3d}/{args.epochs} - "
            f"Train RND Loss: {train_loss:.6f} | "
            f"Val RND Loss: {val_loss:.6f}"
        )

        is_save = (epoch % args.save_interval == 0) or (epoch == args.epochs)
        if is_save:
            save_path = save_dir / f'rnd_epoch{epoch}.pth'
            tmp_path  = save_path.with_suffix('.tmp')

            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()

            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({
                'epoch': epoch,
                'args' : vars(args),
                'model': cpu_state,
            }, tmp_path)
            tmp_path.replace(save_path)
            tqdm.write(f"  ✓ Saved checkpoint: {save_path.name}")

    print("\n" + "=" * 80)
    print("RND training complete!")
    print(f"Checkpoints saved to: {save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
