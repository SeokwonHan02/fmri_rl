"""
Jackknife Bootstrapped Ensemble DQN (offline RL).

Pipeline
--------
1. Pool & shuffle  : Load all training transitions into one dataset, globally
                     shuffle at the transition level.
2. K-fold split    : Divide the shuffled pool into K=20 non-overlapping,
                     equal-size folds.
3. Leave-one-out   : Model i trains on the K-1 folds that exclude fold i.
                     Implemented via index masking on the shared base dataset
                     — zero extra memory for duplicated transitions.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import glob
import argparse
from pathlib import Path
from tqdm import tqdm

from dataset import OfflineRLDataset
from torch.utils.data import DataLoader, Subset
from model.dqn import DQN


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_train_test_files(data_dir: str, subject: str,
                          train_indices: list[int], test_file_idx: int):
    """
    Resolve npz file paths from index lists.

    Returns
    -------
    train_files : list[str]
    test_files  : list[str]
    """
    subject_dir = Path(data_dir) / subject
    if not subject_dir.exists():
        raise ValueError(f"Subject directory not found: {subject_dir}")

    npz_files = sorted(glob.glob(str(subject_dir / '*.npz')))
    n = len(npz_files)
    if n == 0:
        raise ValueError(f"No .npz files in {subject_dir}")

    for idx in train_indices:
        if not (0 <= idx < n):
            raise ValueError(f"train_idx {idx} out of range [0, {n-1}]")
    if not (0 <= test_file_idx < n):
        raise ValueError(f"test_file_idx={test_file_idx} out of range [0, {n-1}]")
    if test_file_idx in set(train_indices):
        raise ValueError(f"test_file_idx={test_file_idx} overlaps with train_idx")

    train_files = [npz_files[i] for i in sorted(train_indices)]
    test_files  = [npz_files[test_file_idx]]
    return train_files, test_files


def make_kfolds(total_size: int, k: int, seed: int) -> list[np.ndarray]:
    """
    Globally shuffle [0, total_size) and split into k equal folds.

    Transitions that don't divide evenly are discarded (at most k-1 dropped).

    Returns
    -------
    folds : list of k numpy arrays, each of length (total_size // k)
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_size)

    fold_size = total_size // k
    n_keep    = fold_size * k          # drop the remainder
    dropped   = total_size - n_keep

    if dropped > 0:
        print(f"  [K-fold] Dropped {dropped} transitions to make {k} equal folds "
              f"of {fold_size} each.")

    perm = perm[:n_keep]
    folds = np.split(perm, k)         # list of k arrays, each length fold_size
    return folds


def make_loader(dataset, indices: np.ndarray, batch_size: int,
                num_workers: int, shuffle: bool) -> DataLoader:
    """
    Wrap `dataset` with the given `indices` via torch.utils.data.Subset.
    No data is copied — Subset stores only the index array.
    """
    subset = Subset(dataset, indices.tolist())
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Target network update
# ---------------------------------------------------------------------------

def soft_update(policy_net: torch.nn.Module, target_net: torch.nn.Module, tau: float):
    for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
        tp.data.copy_(tau * pp.data + (1.0 - tau) * tp.data)


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def train_epoch(model, target_model, loader, optimizer, device,
                gamma, use_soft_update, tau, update_freq, step_counter):
    model.train()
    total_loss = total_q = n = 0

    for batch in loader:
        step_counter += 1

        s  = batch['state'].to(device).float() / 255.0
        a  = batch['action'].to(device)
        r  = batch['reward'].to(device).float()
        ns = batch['next_state'].to(device).float() / 255.0
        d  = batch['done'].to(device).float()

        if r.dim() == 2: r = r.squeeze(1)
        if d.dim() == 2: d = d.squeeze(1)

        a_idx = a.argmax(dim=-1) if a.dim() == 2 else a.long()

        q_all = model(s)                                           # (B, A)
        q_sa  = q_all.gather(1, a_idx.unsqueeze(1)).squeeze(1)    # (B,)

        with torch.no_grad():
            next_q = target_model(ns).max(dim=1)[0]               # (B,)
            target = r + gamma * next_q * (1.0 - d)               # (B,)

        loss = F.smooth_l1_loss(q_sa, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if use_soft_update:
            soft_update(model, target_model, tau)
        elif step_counter % update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        bs          = s.size(0)
        total_loss += loss.item() * bs
        total_q    += q_all.mean().item() * bs
        n          += bs

    return total_loss / n, total_q / n, step_counter


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Human-action accuracy and CE loss on the test set.

    Q-values are Z-score normalised per sample before computing CE,
    so the scale of Q does not affect the metric.

    Returns
    -------
    ce  : float   – mean cross-entropy (lower = model prefers human actions)
    acc : float   – fraction of transitions where argmax Q == human action
    """
    model.eval()
    total_ce = total_correct = n = 0

    for batch in loader:
        s = batch['state'].to(device).float() / 255.0
        a = batch['action'].to(device)

        a_idx = a.argmax(dim=-1) if a.dim() == 2 else a.long()

        q    = model(s)                                       # (B, A)
        z    = (q - q.mean(-1, keepdim=True)) / (q.std(-1, keepdim=True) + 1e-8)

        ce   = F.cross_entropy(z, a_idx, reduction='sum')
        pred = z.argmax(dim=-1)

        total_ce      += ce.item()
        total_correct += (pred == a_idx).sum().item()
        n             += s.size(0)

    return total_ce / n, total_correct / n


# ---------------------------------------------------------------------------
# Train one ensemble member
# ---------------------------------------------------------------------------

def train_single_model(model_idx: int, fold_idx: int,
                       base_dataset, train_indices: np.ndarray,
                       test_loader, device, args, save_dir: Path):
    """
    Train model `model_idx` on `train_indices` (all folds except fold `fold_idx`).

    Parameters
    ----------
    model_idx     : position in the ensemble (0-based)
    fold_idx      : which fold is excluded (== model_idx in leave-one-out)
    base_dataset  : shared OfflineRLDataset (read-only, never copied)
    train_indices : 1-D numpy array of transition indices for this model
    test_loader   : shared DataLoader for the held-out test file
    """
    seed = args.base_seed + model_idx

    print(f"\n{'='*80}")
    print(f"Model {model_idx:2d}/{args.ensemble_size-1}  |  excluded fold={fold_idx}  |  seed={seed}")
    print(f"  Train transitions : {len(train_indices):,}  "
          f"(= {args.ensemble_size-1}/{args.ensemble_size} of pool)")
    print(f"{'='*80}")

    set_seed(seed)

    # ---- DataLoader (zero-copy: Subset wraps the shared base_dataset) -------
    train_loader = make_loader(base_dataset, train_indices,
                               args.batch_size, args.num_workers, shuffle=True)

    # ---- Model --------------------------------------------------------------
    model        = DQN(action_dim=6).to(device)
    target_model = DQN(action_dim=6).to(device)
    target_model.load_state_dict(model.state_dict())
    for p in target_model.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"  Params     : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  LR         : {args.lr:.2e}")
    print(f"  Target upd : "
          f"{'soft τ=' + str(args.tau) if args.use_soft_update else 'hard freq=' + str(args.target_update_freq)}")

    # ---- Training loop ------------------------------------------------------
    step_counter = 0
    summary_rows = []

    for epoch in tqdm(range(1, args.epochs + 1),
                      desc=f"Model {model_idx:02d}", unit="epoch"):

        train_loss, train_q, step_counter = train_epoch(
            model, target_model, train_loader, optimizer, device,
            gamma          = args.gamma,
            use_soft_update= args.use_soft_update,
            tau            = args.tau,
            update_freq    = args.target_update_freq,
            step_counter   = step_counter,
        )

        test_ce, test_acc = evaluate(model, test_loader, device)

        summary_rows.append({'epoch': epoch, 'test_ce': test_ce, 'test_acc': test_acc})

        tqdm.write(
            f"  [Model {model_idx:02d}] Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss={train_loss:.4f}  Q={train_q:.3f} | "
            f"Test  CE={test_ce:.4f}  Acc={test_acc:.4f}"
        )

        # Checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt = save_dir / f'model_{model_idx:02d}_epoch{epoch:03d}.pth'
            tmp  = ckpt.with_suffix('.tmp')
            if device.type == 'mps':   torch.mps.synchronize()
            elif device.type == 'cuda': torch.cuda.synchronize()
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, tmp)
            tmp.replace(ckpt)

    last = summary_rows[-1]
    print(f"  Model {model_idx:02d} done | "
          f"Test CE={last['test_ce']:.4f}  Acc={last['test_acc']:.4f}")

    return summary_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Jackknife Bootstrapped Ensemble DQN (offline RL)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ---------------------------------------------------------------
    parser.add_argument('--data_dir', type=str,
                        default='/Users/seokwon/research/fMRI_RL/processed_data_frameskip_4')
    parser.add_argument('--subject', type=str, default='sub_1')
    parser.add_argument('--train_idx', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4, 7, 8, 9, 10, 11],
                        help='File indices for training (space-separated). '
                             'Example: --train_idx 0 1 2 3 4 7 8 9')
    parser.add_argument('--test_file_idx', type=int, default=6,
                        help='File index for test evaluation')

    # ---- Ensemble -----------------------------------------------------------
    parser.add_argument('--ensemble_size', type=int, default=20,
                        help='K: number of folds = number of ensemble members')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Global shuffle seed and base model seed '
                             '(model i uses base_seed + i)')

    # ---- Training -----------------------------------------------------------
    parser.add_argument('--epochs',     type=int,   default=25)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--gamma',      type=float, default=0.99)

    # ---- Target network -----------------------------------------------------
    parser.add_argument('--use_soft_update',    action='store_true', default=False,
                        help='Use soft (Polyak) target update instead of hard copy')
    parser.add_argument('--tau',                type=float, default=0.005,
                        help='Soft update rate (only with --use_soft_update)')
    parser.add_argument('--target_update_freq', type=int,   default=1000,
                        help='Hard update frequency in gradient steps')

    # ---- System -------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 1. Resolve file paths
    # ------------------------------------------------------------------
    train_files, test_files = load_train_test_files(
        args.data_dir, args.subject, args.train_idx, args.test_file_idx
    )
    print(f"\nTrain files ({len(train_files)}):")
    for f in train_files:
        print(f"  {Path(f).name}")
    print(f"Test  file : {Path(test_files[0]).name}")

    # ------------------------------------------------------------------
    # 2. Load ALL training data into a single shared dataset (once)
    # ------------------------------------------------------------------
    print("\nLoading all training transitions into memory (shared pool)...")
    base_dataset = OfflineRLDataset(npz_files=train_files)
    N = len(base_dataset)
    print(f"  Total training transitions: {N:,}")

    # ------------------------------------------------------------------
    # 3. Global shuffle → K equal folds (index-level, no data copy)
    # ------------------------------------------------------------------
    K = args.ensemble_size
    print(f"\nCreating {K} folds (global seed={args.base_seed})...")
    folds = make_kfolds(N, K, seed=args.base_seed)
    fold_size = len(folds[0])
    print(f"  Fold size       : {fold_size:,} transitions each")
    print(f"  Train per model : {fold_size * (K-1):,} transitions "
          f"({(K-1)/K*100:.1f}% of pool)")
    print(f"  Excluded / model: {fold_size:,} transitions "
          f"({1/K*100:.1f}% of pool)")

    # ------------------------------------------------------------------
    # 4. Shared test loader
    # ------------------------------------------------------------------
    print(f"\nLoading test data...")
    test_dataset = OfflineRLDataset(npz_files=test_files)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"  Test transitions: {len(test_dataset):,}")

    # ------------------------------------------------------------------
    # 5. Train K ensemble members sequentially
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"Jackknife Ensemble: {K} models, leave-one-fold-out")
    print(f"{'='*80}")

    all_summaries: dict[int, list[dict]] = {}

    for model_idx in range(K):
        # Concatenate all folds except fold model_idx — zero data copy
        train_indices = np.concatenate(
            [folds[j] for j in range(K) if j != model_idx]
        )

        summary = train_single_model(
            model_idx    = model_idx,
            fold_idx     = model_idx,
            base_dataset = base_dataset,
            train_indices= train_indices,
            test_loader  = test_loader,
            device       = device,
            args         = args,
            save_dir     = save_dir,
        )
        all_summaries[model_idx] = summary

    # ------------------------------------------------------------------
    # 6. Final summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("ENSEMBLE COMPLETE — final-epoch metrics per model")
    print(f"{'='*80}")
    print(f"  {'Model':>5}  {'Test CE':>8}  {'Test Acc':>9}")
    print(f"  {'-'*28}")
    for idx, rows in all_summaries.items():
        last = rows[-1]
        print(f"  {idx:5d}  {last['test_ce']:8.4f}  {last['test_acc']:9.4f}")

    ces  = [r[-1]['test_ce']  for r in all_summaries.values()]
    accs = [r[-1]['test_acc'] for r in all_summaries.values()]
    print(f"  {'-'*28}")
    print(f"  {'Mean':>5}  {np.mean(ces):8.4f}  {np.mean(accs):9.4f}")
    print(f"  {'Std':>5}  {np.std(ces):8.4f}  {np.std(accs):9.4f}")
    print(f"\nAll checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
