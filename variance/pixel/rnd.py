"""
rnd.py

Random Network Distillation (RND) 모델 정의 및 학습.

입력: 4-frame stack의 마지막 frame (1채널, 84x84), /255.0 정규화.
학습 split: expanding window — test_file_idx 이전 파일들만 사용.

체크포인트:
  <save_dir>/epoch_{n}.pth  - 매 epoch 저장
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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import OfflineRLDataset


# ─── Model ───────────────────────────────────────────────────────────────────

class TargetNetwork(nn.Module):
    """
    Fixed random target network (1-channel input: last frame only).
    Weights are randomly initialized and NEVER updated.
    """
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4), nn.LeakyReLU(0.01),  # → (32, 20, 20)
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.LeakyReLU(0.01), # → (64,  9,  9)
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.LeakyReLU(0.01), # → (64,  7,  7)
            nn.Flatten(),                                                    # → 3136
            nn.Linear(3136, feature_dim),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictorNetwork(nn.Module):
    """
    Trainable predictor network (1-channel input: last frame only).
    Extra hidden layer gives the predictor more capacity than the target
    (following the original RND paper).
    """
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.LeakyReLU(0.01),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RND(nn.Module):
    """
    RND module.

    Input: (B, 1, 84, 84) float [0, 1]  (last frame / 255.0)
    Forward returns (pred_feat, target_feat).
    Novelty score = MSE(pred_feat, target_feat) per frame.
    """
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.target    = TargetNetwork(feature_dim)
        self.predictor = PredictorNetwork(feature_dim)

    def forward(self, x: torch.Tensor):
        """x: (B, 1, 84, 84) float [0, 1]. Returns (pred_feat, target_feat)."""
        target_feat = self.target(x)
        pred_feat   = self.predictor(x)
        return pred_feat, target_feat

    @torch.no_grad()
    def novelty_score(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 84, 84) float [0, 1]. Returns per-frame MSE (B,)."""
        self.eval()
        pred_feat, target_feat = self.forward(x)
        return ((pred_feat - target_feat) ** 2).mean(dim=1)


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

def _eval_loss(model: RND, loader: DataLoader, device: torch.device) -> float:
    """Compute mean prediction error MSE on a DataLoader (no grad)."""
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch['state'][:, -1:, :, :].to(device).float() / 255.0
            pred_feat, target_feat = model(x)
            total += ((pred_feat - target_feat) ** 2).mean(dim=1).sum().item()
            n     += x.size(0)
    return total / n


def train(model: RND, loader: DataLoader, test_loader: DataLoader,
          device: torch.device, epochs: int, lr: float, save_dir: Path,
          loss_scale: float = 1000.0, save_iter: int = 100):
    """
    Train only the predictor network.
    Saves epoch_{n}.pth every save_iter epochs (and always at the last epoch).
    loss_scale: multiply loss values for display only (raw values stored in checkpoint).
    """
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    history = []

    for epoch in range(1, epochs + 1):
        model.predictor.train()
        model.target.eval()

        epoch_loss = 0.0
        n_batches  = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x = batch['state'][:, -1:, :, :].to(device).float() / 255.0  # (B, 1, 84, 84)

            optimizer.zero_grad()
            pred_feat, target_feat = model(x)
            loss = ((pred_feat - target_feat.detach()) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.predictor.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_train = epoch_loss / n_batches
        avg_test  = _eval_loss(model, test_loader, device)

        # Restore train mode after eval
        model.predictor.train()
        model.target.eval()

        history.append(dict(epoch=epoch, train=avg_train, test=avg_test))
        scheduler.step(avg_train)

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train={avg_train * loss_scale:.4f}  "
              f"test={avg_test  * loss_scale:.4f}  "
              f"(×{loss_scale:.0f})")

        if epoch % save_iter == 0 or epoch == epochs:
            torch.save({
                'epoch':       epoch,
                'predictor':   model.predictor.state_dict(),
                'target':      model.target.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'train_loss':  avg_train,
                'test_loss':   avg_test,
                'feature_dim': model.feature_dim,
            }, save_dir / f'epoch_{epoch}.pth')

    # Training curve (scaled for readability)
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs_ax = [h['epoch'] for h in history]
    ax.plot(epochs_ax, [h['train'] * loss_scale for h in history],
            color='steelblue', label='Train')
    ax.plot(epochs_ax, [h['test']  * loss_scale for h in history],
            color='coral',     label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Prediction Error (MSE ×{loss_scale:.0f})')
    ax.set_title('RND Training Loss')
    ax.legend(); ax.grid(alpha=0.3)
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
    parser = argparse.ArgumentParser(description='RND Pixel Novelty — Training')

    # Data
    parser.add_argument('--data_dir',      type=str,
                        default=str(_ROOT_DIR / 'processed_data_frameskip_4'))
    parser.add_argument('--subject',       type=str,  default='sub_1')
    parser.add_argument('--test_file_idx', type=int,  default=10,
                        help='Index of test file; files 0..test_file_idx-1 used for training')

    # Model
    parser.add_argument('--feature_dim', type=int, default=512)

    # Training
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--save_iter',   type=int,   default=100,
                        help='체크포인트 저장 주기 (epoch 단위); 마지막 epoch은 항상 저장')
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int,   default=0)

    # Output
    parser.add_argument('--save_dir', type=str,
                        default=str(_SCRIPT_DIR / 'rnd_results'))

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

    print(f"Device      : {device}")
    print(f"Save dir    : {save_dir}")
    print(f"Feature dim : {args.feature_dim}  epochs={args.epochs}")

    print("\n" + "="*60)
    print("Loading data")
    print("="*60)
    train_loader, test_loader = build_loaders(args)

    print("\n" + "="*60)
    print("Initializing RND")
    print("="*60)
    model = RND(feature_dim=args.feature_dim).to(device)
    n_target    = sum(p.numel() for p in model.target.parameters())
    n_predictor = sum(p.numel() for p in model.predictor.parameters())
    print(f"  Target    parameters (frozen) : {n_target:,}")
    print(f"  Predictor parameters (trained): {n_predictor:,}")
    print(f"  Input                         : last frame (1-channel, 84×84) / 255")

    print("\n" + "="*60)
    print("Training predictor")
    print("="*60)
    train(model, train_loader, test_loader, device, args.epochs, args.lr, save_dir,
          save_iter=args.save_iter)

    print(f"\nDone. Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
