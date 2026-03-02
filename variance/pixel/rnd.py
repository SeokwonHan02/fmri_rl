"""
rnd.py

Random Network Distillation (RND) 기반 pixel novelty 측정.

고정된 랜덤 타겟 네트워크와 훈련되는 예측 네트워크 간의
예측 오류(MSE)를 pixel-level novelty score로 사용.
예측 오류가 클수록 → 훈련 분포와 거리가 먼(OOD) frame.

학습 split(val/test 제외한 모든 파일)으로 predictor를 훈련.

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


# ─── Model ───────────────────────────────────────────────────────────────────

class TargetNetwork(nn.Module):
    """
    Fixed random target network.
    Produces a feature embedding for a given frame stack.
    Weights are randomly initialized and NEVER updated.
    """
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.LeakyReLU(0.01),  # → (32, 20, 20)
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.LeakyReLU(0.01), # → (64,  9,  9)
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.LeakyReLU(0.01), # → (64,  7,  7)
            nn.Flatten(),                                                    # → 3136
            nn.Linear(3136, feature_dim),
        )
        # Freeze all weights — never call optimizer.step() on this
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictorNetwork(nn.Module):
    """
    Trainable predictor network.
    Learns to predict the target network's output for in-distribution frames.
    High prediction error → out-of-distribution (novel) frame.

    Extra hidden layer gives the predictor more capacity than the target,
    following the original RND paper.
    """
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.LeakyReLU(0.01),  # → (32, 20, 20)
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.LeakyReLU(0.01), # → (64,  9,  9)
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.LeakyReLU(0.01), # → (64,  7,  7)
            nn.Flatten(),                                                    # → 3136
            nn.Linear(3136, 512),     nn.ReLU(),
            nn.Linear(512,  feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RND(nn.Module):
    """
    RND module: wraps target + predictor.

    Novelty score = MSE( predictor(x), target(x).detach() )
    averaged over the feature dimension, per frame.
    """
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.target    = TargetNetwork(feature_dim)
        self.predictor = PredictorNetwork(feature_dim)

    def forward(self, x: torch.Tensor):
        """Returns (pred_features, target_features). Both (B, feature_dim)."""
        target_feat = self.target(x)          # fixed
        pred_feat   = self.predictor(x)       # trainable
        return pred_feat, target_feat

    @torch.no_grad()
    def novelty_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame prediction error (MSE over feature dim). Returns (B,)."""
        self.eval()
        pred_feat, target_feat = self.forward(x)
        return ((pred_feat - target_feat) ** 2).mean(dim=1)


# ─── Data helpers ─────────────────────────────────────────────────────────────

def build_loaders(args):
    """Returns (train_loader, val_loader, test_loader, split_info)."""
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

def train(model: RND, loader: DataLoader, device: torch.device,
          epochs: int, lr: float, save_dir: Path):
    """
    Train only the predictor network.
    Objective: minimize MSE between predictor and target embeddings on train data.
    """
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True)

    best_loss = float('inf')
    history   = []

    for epoch in range(1, epochs + 1):
        model.predictor.train()
        model.target.eval()   # target always in eval, never updated

        epoch_loss = 0.0
        n_batches  = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x = batch['state'].to(device).float() / 255.0  # (B, 4, 84, 84)
            optimizer.zero_grad()

            pred_feat, target_feat = model(x)
            # Prediction error — target features must NOT receive gradients
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
        novelty  : (N,)           - per-frame prediction error (MSE over feature dim)
        actions  : (N,)           - action index taken
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
    ax.set_xlabel('Prediction Error MSE (pixel novelty)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_novelty_frames(indices, frames: np.ndarray,
                        novelty: np.ndarray, actions: np.ndarray,
                        title: str, save_path: Path,
                        model: RND, device: torch.device):
    """
    For each selected frame:
      col 1 – last grayscale frame
      col 2 – feature prediction error bar chart (per feature dimension group)
    """
    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(14, n * 2.8))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    axes[0, 0].set_title('Frame (last channel)', fontsize=9, fontweight='bold')
    axes[0, 1].set_title('Prediction Error per Feature Group', fontsize=9, fontweight='bold')

    model.eval()
    n_groups = 16  # visualize feature errors in 16 groups
    with torch.no_grad():
        for row, idx in enumerate(indices):
            raw = frames[idx]                                        # (4, 84, 84) uint8
            x   = torch.from_numpy(raw).float().unsqueeze(0) / 255.0
            x   = x.to(device)

            pred_feat, target_feat = model(x)                        # (1, feature_dim)
            err_per_feat = ((pred_feat - target_feat) ** 2)[0].cpu().numpy()  # (feature_dim,)
            group_size   = len(err_per_feat) // n_groups
            err_grouped  = err_per_feat[:n_groups * group_size].reshape(n_groups, -1).mean(1)

            act = actions[idx]
            nov = novelty[idx]

            # Frame image
            axes[row, 0].imshow(raw[-1], cmap='gray', vmin=0, vmax=255)
            axes[row, 0].set_title(
                f"Frame #{idx} | Action: {ACTION_NAMES[act]}\nNovelty: {nov:.5f}",
                fontsize=7)
            axes[row, 0].axis('off')

            # Error bar chart
            x_pos = np.arange(n_groups)
            axes[row, 1].bar(x_pos, err_grouped, color='steelblue', alpha=0.85)
            axes[row, 1].axhline(err_grouped.mean(), color='red', linestyle='--',
                                  linewidth=1.2, label=f'mean={err_grouped.mean():.4f}')
            axes[row, 1].set_xlabel('Feature group', fontsize=7)
            axes[row, 1].set_ylabel('MSE', fontsize=7)
            axes[row, 1].set_title(f'Total error = {nov:.5f}', fontsize=8)
            axes[row, 1].legend(fontsize=7)
            axes[row, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def print_stats(label: str, novelty: np.ndarray):
    print(f"\n[{label}] Pixel Novelty (RND Prediction Error):")
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
    parser.add_argument('--val_file_idx',  type=int,  default=9,
                        help='Index of validation file (0-based)')
    parser.add_argument('--test_file_idx', type=int,  default=10,
                        help='Index of test file (0-based)')

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

    # Reproducibility
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
    train_loader, val_loader, test_loader, split_info = build_loaders(args)

    # ── Model ───────────────────────────────
    print("\n" + "="*60)
    print("Initializing RND")
    print("="*60)
    model = RND(feature_dim=args.feature_dim).to(device)
    n_target    = sum(p.numel() for p in model.target.parameters())
    n_predictor = sum(p.numel() for p in model.predictor.parameters())
    print(f"  Target    parameters (frozen) : {n_target:,}")
    print(f"  Predictor parameters (trained): {n_predictor:,}")

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
            title     = f'RND Pixel Novelty — {split_name.upper()}',
            save_path = save_dir / f'{split_name}_novelty_histogram.png',
        )

        # Top-K high/low novelty frames
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
