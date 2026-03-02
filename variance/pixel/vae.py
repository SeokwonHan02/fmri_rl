"""
vae.py

Variational Autoencoder 기반 pixel novelty 측정.

학습 split(val/test 제외한 모든 파일)으로 VAE를 훈련하고,
각 frame의 재건 오류(MSE)를 pixel-level novelty score로 사용.

결과:
  <save_dir>/model.pth                   - 학습된 VAE 체크포인트
  <save_dir>/{split}_novelty.npz        - 분할별 novelty 점수
  <save_dir>/{split}_novelty_histogram.png
  <save_dir>/{split}_recon_frames.png   - 원본/재건/오류맵 비교
  <save_dir>/{split}_high_novelty_frames.png
  <save_dir>/{split}_low_novelty_frames.png
"""

import sys
import glob
import argparse
import random
from pathlib import Path

# ─── Path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent        # fMRI_RL/variance/pixel/
_ROOT_DIR   = _SCRIPT_DIR.parent.parent    # fMRI_RL/
sys.path.insert(0, str(_ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import OfflineRLDataset


# ─── Model ───────────────────────────────────────────────────────────────────

class VAEEncoder(nn.Module):
    """Conv encoder: (4, 84, 84) → (mu, logvar) each of shape (latent_dim,)"""
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),  # → (32, 20, 20)
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), # → (64,  9,  9)
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(), # → (64,  7,  7)
            nn.Flatten(),                                           # → 3136
        )
        self.fc     = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
        self.mu     = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        h = self.fc(self.conv(x))
        return self.mu(h), self.logvar(h)


class VAEDecoder(nn.Module):
    """Conv decoder: (latent_dim,) → (4, 84, 84) in [0, 1]"""
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 3136),       nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),  # → (64, 9, 9)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2), nn.ReLU(),  # → (32, 20, 20)
            nn.ConvTranspose2d(32,  4, kernel_size=8, stride=4), nn.Sigmoid(), # → (4, 84, 84)
        )

    def forward(self, z):
        return self.deconv(self.fc(z))


class VAE(nn.Module):
    """
    Variational Autoencoder for 4-stacked Atari frames.

    Novelty score = per-frame reconstruction MSE (averaged over C×H×W).
    High score → unfamiliar / out-of-distribution frame.
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu  # deterministic at eval time

    def forward(self, x: torch.Tensor):
        """Returns (recon, mu, logvar)."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def novelty_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame reconstruction MSE. x ∈ [0, 1]. Returns (B,)."""
        self.eval()
        recon, _, _ = self.forward(x)
        return ((recon - x) ** 2).mean(dim=(1, 2, 3))


# ─── Loss ────────────────────────────────────────────────────────────────────

def vae_loss(recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0):
    """
    ELBO loss = MSE reconstruction + beta * KL divergence.

    recon_loss: mean over batch of sum over pixels (matches standard ELBO scaling)
    kl_loss:    mean over batch of KL per latent dimension
    """
    # Per-sample sum over pixels, then mean over batch
    recon_loss = F.mse_loss(recon, x, reduction='none').sum(dim=(1, 2, 3)).mean()
    # KL: -0.5 * Σ(1 + logvar - μ² - exp(logvar))
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    total = recon_loss + beta * kl_loss
    return total, recon_loss.item(), kl_loss.item()


# ─── Data helpers ─────────────────────────────────────────────────────────────

def build_loaders(args):
    """
    Returns (train_loader, val_loader, test_loader, split_info).
    Train: all files except val and test.
    """
    subject_dir = Path(args.data_dir) / args.subject
    npz_files   = sorted(glob.glob(str(subject_dir / '*.npz')))
    n = len(npz_files)

    print(f"Available files ({n}):")
    for i, f in enumerate(npz_files):
        tag = (' ← VAL'  if i == args.val_file_idx  else
               ' ← TEST' if i == args.test_file_idx else '')
        print(f"  [{i:2d}] {Path(f).name}{tag}")

    exclude     = {args.val_file_idx, args.test_file_idx}
    train_files = [f for i, f in enumerate(npz_files) if i not in exclude]
    val_files   = [npz_files[args.val_file_idx]]
    test_files  = [npz_files[args.test_file_idx]]

    pin = torch.cuda.is_available()

    print(f"\nLoading train data ({len(train_files)} files)...")
    train_ds = OfflineRLDataset(npz_files=train_files)
    print(f"Loading val data...")
    val_ds   = OfflineRLDataset(npz_files=val_files)
    print(f"Loading test data...")
    test_ds  = OfflineRLDataset(npz_files=test_files)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    split_info = dict(
        train_files=train_files,
        val_file=val_files[0],
        test_file=test_files[0],
    )
    return train_loader, val_loader, test_loader, split_info


# ─── Training ─────────────────────────────────────────────────────────────────

def train(model: VAE, loader: DataLoader, device: torch.device,
          epochs: int, lr: float, beta: float, save_dir: Path):
    """Train the VAE and save checkpoint after each epoch (keep best)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True)

    best_loss = float('inf')
    history   = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = recon_sum = kl_sum = 0.0
        n_batches  = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x = batch['state'].to(device).float() / 255.0  # (B, 4, 84, 84)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            recon_sum  += recon_loss
            kl_sum     += kl_loss
            n_batches  += 1

        avg_total = total_loss / n_batches
        avg_recon = recon_sum  / n_batches
        avg_kl    = kl_sum     / n_batches
        history.append(dict(epoch=epoch, total=avg_total, recon=avg_recon, kl=avg_kl))
        scheduler.step(avg_total)

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"total={avg_total:.4f}  recon={avg_recon:.4f}  kl={avg_kl:.4f}")

        # Save best checkpoint
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'epoch':      epoch,
                'model':      model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'loss':       avg_total,
                'latent_dim': model.latent_dim,
                'beta':       beta,
            }, save_dir / 'model.pth')
            print(f"    → Best model saved (loss={best_loss:.4f})")

    # Plot training curve
    epochs_axis = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, color in zip(axes, ['total', 'recon', 'kl'],
                               ['steelblue', 'coral', 'mediumseagreen']):
        ax.plot(epochs_axis, [h[key] for h in history], color=color)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title(key.capitalize() + ' Loss'); ax.grid(alpha=0.3)
    plt.suptitle('VAE Training Loss', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curve.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nTraining curve saved → {save_dir / 'training_curve.png'}")

    return history


# ─── Novelty computation ──────────────────────────────────────────────────────

def compute_novelty(model: VAE, loader: DataLoader, device: torch.device):
    """
    Returns:
        novelty  : (N,)          - per-frame reconstruction MSE
        actions  : (N,)          - action index taken
        frames   : (N, 4, 84, 84) uint8
    """
    model.eval()
    novelty_list = []
    actions_list = []
    frames_list  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Computing novelty"):
            x      = batch['state'].to(device).float() / 255.0
            action = batch['action'].to(device)
            action_idx = action.argmax(dim=-1) if action.dim() == 2 else action.long()

            scores = model.novelty_score(x)  # (B,)

            novelty_list.append(scores.cpu().numpy())
            actions_list.append(action_idx.cpu().numpy())
            frames_list.append(batch['state'].numpy())

    return (
        np.concatenate(novelty_list),   # (N,)
        np.concatenate(actions_list),   # (N,)
        np.concatenate(frames_list),    # (N, 4, 84, 84)
    )


# ─── Visualization ────────────────────────────────────────────────────────────

ACTION_NAMES = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHT+FIRE', 'LEFT+FIRE']


def plot_histogram(novelty: np.ndarray, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(novelty, bins=100, color='steelblue', edgecolor='white', linewidth=0.4)
    p5  = np.percentile(novelty, 5)
    p95 = np.percentile(novelty, 95)
    ax.axvline(p95, color='red',   linestyle='--', label=f'95th pct ({p95:.5f})')
    ax.axvline(p5,  color='green', linestyle='--', label=f'5th  pct ({p5:.5f})')
    ax.set_xlabel('Reconstruction MSE (pixel novelty)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_recon_comparison(model: VAE, frames: np.ndarray, indices,
                          novelty: np.ndarray, actions: np.ndarray,
                          title: str, save_path: Path, device: torch.device):
    """
    For each selected frame, show:
      col 1 – original last channel (grayscale)
      col 2 – reconstructed last channel
      col 3 – per-pixel error map (absolute diff)
    """
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.8))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    col_titles = ['Original', 'Reconstructed', 'Error Map']
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=10, fontweight='bold')

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            raw   = frames[idx]                                    # (4, 84, 84) uint8
            x     = torch.from_numpy(raw).float().unsqueeze(0) / 255.0
            x     = x.to(device)
            recon, _, _ = model(x)                                 # (1, 4, 84, 84)
            recon_np = recon.squeeze(0).cpu().numpy()              # (4, 84, 84) float

            orig_img  = raw[-1]                                    # last frame, (84, 84)
            recon_img = recon_np[-1]
            err_img   = np.abs(recon_img - orig_img / 255.0)

            act = actions[idx]
            nov = novelty[idx]

            row_label = (f"Frame #{idx} | Action: {ACTION_NAMES[act]}\n"
                         f"Novelty (MSE): {nov:.5f}")

            axes[row, 0].imshow(orig_img,  cmap='gray', vmin=0, vmax=255)
            axes[row, 1].imshow(recon_img, cmap='gray', vmin=0, vmax=1)
            im = axes[row, 2].imshow(err_img, cmap='hot', vmin=0, vmax=0.5)
            plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

            axes[row, 0].set_ylabel(row_label, fontsize=7, rotation=0,
                                    labelpad=80, va='center')
            for j in range(3):
                axes[row, j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def print_stats(label: str, novelty: np.ndarray):
    print(f"\n[{label}] Pixel Novelty (Reconstruction MSE):")
    print(f"  N frames : {len(novelty):,}")
    print(f"  Mean     : {novelty.mean():.5f}")
    print(f"  Std      : {novelty.std():.5f}")
    print(f"  Min      : {novelty.min():.5f}")
    print(f"  Max      : {novelty.max():.5f}")
    print(f"  Median   : {np.median(novelty):.5f}")
    print(f"  95th pct : {np.percentile(novelty, 95):.5f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def get_device(preferred: str) -> torch.device:
    if preferred == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    if preferred == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if preferred not in ('mps', 'cuda', 'cpu'):
        # auto
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='VAE Pixel Novelty')

    # Data
    parser.add_argument('--data_dir',      type=str,
                        default=str(_ROOT_DIR / 'processed_data_frameskip_4'))
    parser.add_argument('--subject',       type=str,  default='sub_1')
    parser.add_argument('--val_file_idx',  type=int,  default=9,
                        help='Index of validation file (0-based)')
    parser.add_argument('--test_file_idx', type=int,  default=10,
                        help='Index of test file (0-based)')

    # Model
    parser.add_argument('--latent_dim', type=int,   default=256)
    parser.add_argument('--beta',       type=float, default=1.0,
                        help='KL weight in ELBO loss')

    # Training
    parser.add_argument('--epochs',      type=int,   default=20)
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int,   default=0)

    # Output
    parser.add_argument('--save_dir', type=str,
                        default=str(_SCRIPT_DIR / 'vae_results'))
    parser.add_argument('--top_k',    type=int, default=10)

    # System
    parser.add_argument('--device', type=str, default='auto',
                        help='auto | mps | cuda | cpu')
    parser.add_argument('--seed',   type=int, default=42)

    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device   = get_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device   : {device}")
    print(f"Save dir : {save_dir}")
    print(f"Latent   : {args.latent_dim}  beta={args.beta}  epochs={args.epochs}")

    # ── Data ────────────────────────────────
    print("\n" + "="*60)
    print("Loading data")
    print("="*60)
    train_loader, val_loader, test_loader, split_info = build_loaders(args)

    # ── Model ───────────────────────────────
    print("\n" + "="*60)
    print("Initializing VAE")
    print("="*60)
    model = VAE(latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Train ───────────────────────────────
    print("\n" + "="*60)
    print("Training")
    print("="*60)
    train(model, train_loader, device, args.epochs, args.lr, args.beta, save_dir)

    # Load best checkpoint
    ckpt = torch.load(save_dir / 'model.pth', map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"\nLoaded best checkpoint (epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f})")

    # ── Evaluate splits ─────────────────────
    splits = [
        ('train', train_loader),
        ('val',   val_loader),
        ('test',  test_loader),
    ]

    for split_name, loader in splits:
        print(f"\n{'='*60}")
        print(f"  Split: {split_name.upper()}")
        print(f"{'='*60}")

        novelty, actions, frames = compute_novelty(model, loader, device)
        print_stats(split_name, novelty)

        # Save novelty scores
        npz_path = save_dir / f'{split_name}_novelty.npz'
        np.savez(npz_path, novelty=novelty, actions=actions)
        print(f"\n  Saved arrays → {npz_path.name}")

        # Histogram
        plot_histogram(
            novelty,
            title     = f'VAE Pixel Novelty — {split_name.upper()}',
            save_path = save_dir / f'{split_name}_novelty_histogram.png',
        )

        # Top-K high/low
        sorted_idx   = np.argsort(novelty)
        low_indices  = sorted_idx[:args.top_k]
        high_indices = sorted_idx[-args.top_k:][::-1]

        # Reconstruction comparison (high novelty)
        plot_recon_comparison(
            model, frames, high_indices, novelty, actions,
            title     = f'HIGH Novelty Frames — {split_name.upper()} (top {args.top_k})',
            save_path = save_dir / f'{split_name}_high_novelty_frames.png',
            device    = device,
        )
        # Reconstruction comparison (low novelty)
        plot_recon_comparison(
            model, frames, low_indices, novelty, actions,
            title     = f'LOW Novelty Frames — {split_name.upper()} (bottom {args.top_k})',
            save_path = save_dir / f'{split_name}_low_novelty_frames.png',
            device    = device,
        )

    print(f"\nAll results saved to: {save_dir}")


if __name__ == '__main__':
    main()
