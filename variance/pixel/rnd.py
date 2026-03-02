"""
rnd.py

Random Network Distillation (RND) 기반 pixel novelty 측정.

고정된 랜덤 타겟 네트워크와 훈련되는 예측 네트워크 간의
예측 오류(MSE)를 pixel-level novelty score로 사용.
예측 오류가 클수록 → 훈련 분포와 거리가 먼(OOD) frame.

입력: 4-frame stack의 마지막 frame (1채널, 84x84), /255.0 정규화.
학습 split: expanding window — test_file_idx 이전 파일들만 사용.

결과:
  <save_dir>/model.pth                   - 학습된 predictor 체크포인트
  <save_dir>/{split}_novelty.npz        - 분할별 novelty 점수
  <save_dir>/{split}_novelty_histogram.png
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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import OfflineRLDataset
from rnd_visualize import print_stats, plot_histogram, plot_novelty_frames


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
        tag = (' ← TEST'  if i == args.test_file_idx else
               ' ← TRAIN' if i < args.test_file_idx  else ' ← (excluded)')
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

def train(model: RND, loader: DataLoader, device: torch.device,
          epochs: int, lr: float, save_dir: Path):
    """
    Train only the predictor network.

    Per batch:
      1. Extract last frame → x: (B, 1, 84, 84) float [0, 1]
      2. Forward → (pred_feat, target_feat)
      3. Loss = MSE(pred_feat, target_feat.detach())
    """
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True)

    best_loss = float('inf')
    history   = []

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

        avg_loss = epoch_loss / n_batches
        history.append(dict(epoch=epoch, loss=avg_loss))
        scheduler.step(avg_loss)

        print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch':       epoch,
                'predictor':   model.predictor.state_dict(),
                'target':      model.target.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'loss':        avg_loss,
                'feature_dim': model.feature_dim,
            }, save_dir / 'model.pth')
            print(f"    → Best model saved (loss={best_loss:.6f})")

    # Plot training curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([h['epoch'] for h in history],
            [h['loss']  for h in history], color='steelblue')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Prediction Error (MSE)')
    ax.set_title('RND Training Loss'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curve.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nTraining curve saved → {save_dir / 'training_curve.png'}")

    return history


# ─── Novelty computation ──────────────────────────────────────────────────────

def compute_novelty(model: RND, loader: DataLoader, device: torch.device):
    """
    Returns:
        novelty  : (N,)           - per-frame MSE prediction error
        actions  : (N,)           - action index taken
        frames   : (N, 4, 84, 84) uint8 - full 4-frame stack (for visualization)
    """
    model.eval()
    novelty_list = []
    actions_list = []
    frames_list  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Computing novelty"):
            x = batch['state'][:, -1:, :, :].to(device).float() / 255.0  # (B, 1, 84, 84)
            action = batch['action'].to(device)
            action_idx = action.argmax(dim=-1) if action.dim() == 2 else action.long()

            scores = model.novelty_score(x)  # (B,)

            novelty_list.append(scores.cpu().numpy())
            actions_list.append(action_idx.cpu().numpy())
            frames_list.append(batch['state'].numpy())  # full stack for visualization

    return (
        np.concatenate(novelty_list),   # (N,)
        np.concatenate(actions_list),   # (N,)
        np.concatenate(frames_list),    # (N, 4, 84, 84)
    )


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
    parser = argparse.ArgumentParser(description='RND Pixel Novelty')

    # Data
    parser.add_argument('--data_dir',      type=str,
                        default=str(_ROOT_DIR / 'processed_data_frameskip_4'))
    parser.add_argument('--subject',       type=str,  default='sub_1')
    parser.add_argument('--test_file_idx', type=int,  default=10,
                        help='Index of test file; files 0..test_file_idx-1 used for training')

    # Model
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='Output dimension of target and predictor networks')

    # Training
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int,   default=0)

    # Output
    parser.add_argument('--save_dir', type=str,
                        default=str(_SCRIPT_DIR / 'rnd_results'))
    parser.add_argument('--top_k',    type=int, default=10)

    # System
    parser.add_argument('--device', type=str, default='auto',
                        help='auto | mps | cuda | cpu')
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

    # ── Data ────────────────────────────────
    print("\n" + "="*60)
    print("Loading data")
    print("="*60)
    train_loader, test_loader = build_loaders(args)

    # ── Model ───────────────────────────────
    print("\n" + "="*60)
    print("Initializing RND")
    print("="*60)
    model = RND(feature_dim=args.feature_dim).to(device)
    n_target    = sum(p.numel() for p in model.target.parameters())
    n_predictor = sum(p.numel() for p in model.predictor.parameters())
    print(f"  Target    parameters (frozen) : {n_target:,}")
    print(f"  Predictor parameters (trained): {n_predictor:,}")
    print(f"  Input                         : last frame (1-channel, 84×84) / 255")

    # ── Train ───────────────────────────────
    print("\n" + "="*60)
    print("Training predictor")
    print("="*60)
    train(model, train_loader, device, args.epochs, args.lr, save_dir)

    # Load best checkpoint
    ckpt = torch.load(save_dir / 'model.pth', map_location=device)
    model.predictor.load_state_dict(ckpt['predictor'])
    model.target.load_state_dict(ckpt['target'])
    print(f"\nLoaded best checkpoint (epoch {ckpt['epoch']}, loss={ckpt['loss']:.6f})")

    # ── Evaluate splits ─────────────────────
    splits = [
        ('train', train_loader),
        ('test',  test_loader),
    ]

    for split_name, loader in splits:
        print(f"\n{'='*60}")
        print(f"  Split: {split_name.upper()}")
        print(f"{'='*60}")

        novelty, actions, frames = compute_novelty(model, loader, device)
        print_stats(split_name, novelty)

        npz_path = save_dir / f'{split_name}_novelty.npz'
        np.savez(npz_path, novelty=novelty, actions=actions)
        print(f"\n  Saved arrays → {npz_path.name}")

        plot_histogram(
            novelty,
            title     = f'RND Pixel Novelty — {split_name.upper()}',
            save_path = save_dir / f'{split_name}_novelty_histogram.png',
        )

        sorted_idx   = np.argsort(novelty)
        low_indices  = sorted_idx[:args.top_k]
        high_indices = sorted_idx[-args.top_k:][::-1]

        plot_novelty_frames(
            high_indices, frames, novelty, actions,
            title     = f'HIGH Novelty Frames — {split_name.upper()} (top {args.top_k})',
            save_path = save_dir / f'{split_name}_high_novelty_frames.png',
            model     = model,
            device    = device,
        )
        plot_novelty_frames(
            low_indices, frames, novelty, actions,
            title     = f'LOW Novelty Frames — {split_name.upper()} (bottom {args.top_k})',
            save_path = save_dir / f'{split_name}_low_novelty_frames.png',
            model     = model,
            device    = device,
        )

    print(f"\nAll results saved to: {save_dir}")


if __name__ == '__main__':
    main()
