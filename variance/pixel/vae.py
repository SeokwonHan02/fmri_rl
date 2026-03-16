"""
vae.py

Variational Autoencoder 기반 pixel novelty 측정.

입력: 4-frame stack의 마지막 frame (1채널, 84x84), /255.0 정규화.
학습 split: expanding window — test_file_idx 이전 파일들만 사용.
재건 오류(MSE)를 pixel-level novelty score로 사용.

체크포인트:
  <save_dir>/epoch_{n}.pth
  <save_dir>/training_curve.png
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
    """Conv encoder: (1, 84, 84) → (mu, logvar) each of shape (latent_dim,)"""
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4), nn.ReLU(),  # → (32, 20, 20)
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
    """Conv decoder: (latent_dim,) → (1, 84, 84) in [0, 1]"""
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 3136),       nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),   # → (64, 9, 9)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2), nn.ReLU(),   # → (32, 20, 20)
            nn.ConvTranspose2d(32,  1, kernel_size=8, stride=4), nn.Sigmoid(), # → (1, 84, 84)
        )

    def forward(self, z):
        return self.deconv(self.fc(z))


class VAE(nn.Module):
    """
    VAE for single Atari frame (1-channel, last of 4-stack).

    Novelty score = per-frame reconstruction MSE (averaged over H×W).
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
        """x: (B, 1, 84, 84). Returns (recon, mu, logvar)."""
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

    recon_loss: mean over batch of sum over pixels
    kl_loss:    mean over batch of KL per latent dimension
    """
    recon_loss = F.mse_loss(recon, x, reduction='none').sum(dim=(1, 2, 3)).mean()
    kl_loss    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    total = recon_loss + beta * kl_loss
    return total, recon_loss.item(), kl_loss.item()


# ─── Data helpers ─────────────────────────────────────────────────────────────

def build_loaders(args):
    """Returns (train_loader, test_loader).

    Expanding window: only files with index < test_file_idx are used for training.
    """
    subject_dir = Path(args.data_dir) / args.subject
    npz_files   = sorted(glob.glob(str(subject_dir / '*.npz')))
    n = len(npz_files)

    print(f"Available files ({n}):")
    for i, f in enumerate(npz_files):
        tag = (' ← TEST'      if i == args.test_file_idx else
               ' ← TRAIN'     if i <  args.test_file_idx else
               ' ← (excluded)')
        print(f"  [{i:2d}] {Path(f).name}{tag}")

    train_files = [f for i, f in enumerate(npz_files) if i < args.test_file_idx]
    test_files  = [npz_files[args.test_file_idx]]

    pin = torch.cuda.is_available()

    print(f"\nLoading train data ({len(train_files)} files, expanding window)...")
    train_ds = OfflineRLDataset(npz_files=train_files)
    print(f"Loading test data...")
    test_ds  = OfflineRLDataset(npz_files=test_files)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    return train_loader, test_loader


# ─── Training ─────────────────────────────────────────────────────────────────

def _eval_loss(model: VAE, loader: DataLoader, device: torch.device,
               beta: float) -> tuple:
    """Compute mean total/recon/kl loss on a DataLoader (no grad)."""
    model.eval()
    total_sum = recon_sum = kl_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch['state'][:, -1:, :, :].to(device).float() / 255.0
            recon, mu, logvar = model(x)
            b = x.size(0)
            recon_l = F.mse_loss(recon, x, reduction='none').sum(dim=(1, 2, 3)).sum().item()
            kl_l    = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).sum().item()
            total_sum += (recon_l + beta * kl_l)
            recon_sum += recon_l
            kl_sum    += kl_l
            n         += b
    return total_sum / n, recon_sum / n, kl_sum / n


def train(model: VAE, loader: DataLoader, test_loader: DataLoader,
          device: torch.device, epochs: int, lr: float, beta: float,
          save_dir: Path, save_iter: int = 100):
    """
    Train the VAE. Saves epoch_{n}.pth every save_iter epochs (and always at the last epoch).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = recon_sum = kl_sum = 0.0
        n_batches  = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x = batch['state'][:, -1:, :, :].to(device).float() / 255.0  # (B, 1, 84, 84)
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

        test_total, test_recon, test_kl = _eval_loss(model, test_loader, device, beta)
        model.train()

        history.append(dict(
            epoch=epoch,
            train_total=avg_total, train_recon=avg_recon, train_kl=avg_kl,
            test_total=test_total,  test_recon=test_recon,  test_kl=test_kl,
        ))
        scheduler.step(avg_total)

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train=({avg_total:.2f} / {avg_recon:.2f} / {avg_kl:.2f})  "
              f"test=({test_total:.2f} / {test_recon:.2f} / {test_kl:.2f})  "
              f"[total/recon/kl]")

        if epoch % save_iter == 0 or epoch == epochs:
            torch.save({
                'epoch':       epoch,
                'model':       model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'train_loss':  avg_total,
                'test_loss':   test_total,
                'latent_dim':  model.latent_dim,
                'beta':        beta,
            }, save_dir / f'epoch_{epoch}.pth')

    # Training curve — train vs test for total/recon/kl
    epochs_axis = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, color in zip(axes,
                               ['total', 'recon', 'kl'],
                               ['steelblue', 'coral', 'mediumseagreen']):
        ax.plot(epochs_axis, [h[f'train_{key}'] for h in history],
                color=color, label='Train')
        ax.plot(epochs_axis, [h[f'test_{key}']  for h in history],
                color=color, linestyle='--', label='Test')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title(key.capitalize() + ' Loss')
        ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle('VAE Training Loss', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curve.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Training curve saved → {save_dir / 'training_curve.png'}")

    return history


# ─── Main ─────────────────────────────────────────────────────────────────────

def get_device(preferred: str) -> torch.device:
    if preferred == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    if preferred == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if preferred not in ('mps', 'cuda', 'cpu'):
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='VAE Pixel Novelty — Training')

    # Data
    parser.add_argument('--data_dir',      type=str,
                        default=str(_ROOT_DIR / 'processed_data_frameskip_4'))
    parser.add_argument('--subject',       type=str,  default='sub_1')
    parser.add_argument('--test_file_idx', type=int,  default=10,
                        help='Index of test file; files 0..test_file_idx-1 used for training')

    # Model
    parser.add_argument('--latent_dim', type=int,   default=256)
    parser.add_argument('--beta',       type=float, default=1.0,
                        help='KL weight in ELBO loss')

    # Training
    parser.add_argument('--epochs',      type=int,   default=20)
    parser.add_argument('--save_iter',   type=int,   default=100,
                        help='체크포인트 저장 주기 (epoch 단위); 마지막 epoch은 항상 저장')
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int,   default=0)

    # Output
    parser.add_argument('--save_dir', type=str,
                        default=str(_SCRIPT_DIR / 'vae_results'))

    # System
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed',   type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device   = get_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device   : {device}")
    print(f"Save dir : {save_dir}")
    print(f"Latent   : {args.latent_dim}  beta={args.beta}  epochs={args.epochs}")

    print("\n" + "="*60)
    print("Loading data")
    print("="*60)
    train_loader, test_loader = build_loaders(args)

    print("\n" + "="*60)
    print("Initializing VAE")
    print("="*60)
    model = VAE(latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Input               : last frame (1-channel, 84×84) / 255")

    print("\n" + "="*60)
    print("Training")
    print("="*60)
    train(model, train_loader, test_loader, device, args.epochs, args.lr, args.beta, save_dir,
          save_iter=args.save_iter)

    print(f"\nDone. Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
