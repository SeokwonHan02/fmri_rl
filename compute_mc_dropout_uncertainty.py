"""
compute_mc_dropout_uncertainty.py — MC Dropout-based BC uncertainty extractor

Two modes
---------
train   — train a BCDropout model on human trajectory data with CE loss
extract — run MC Dropout inference to produce per-frame uncertainty regressors

Uncertainty decomposition
-------------------------
  total entropy  = H(mean_probs)                   predictive entropy
  within entropy = mean_t H(probs_t)               aleatoric-like (within-sample)
  across entropy = H(mean) - mean_t H              MC-dropout disagreement
                                                   (NOT true epistemic uncertainty;
                                                    treat as model-derived proxy)
  predictive variance = mean_a Var_t[p_t(a|s)]

Example usage
-------------
  # Train
  python compute_mc_dropout_uncertainty.py --mode train \\
      --subject sub_1 --file-idx 0 1 2 3 4 5 6 7 8 --val-file-idx 9 \\
      --epochs 20 --dropout-p 0.1

  # Extract uncertainty for all runs
  python compute_mc_dropout_uncertainty.py --mode extract \\
      --bc-path checkpoints/bc_dropout/sub_1/final.pth \\
      --subject sub_1 --file-idx 0 1 2 3 4 5 6 7 8 9 10 \\
      --mc-samples 30 --out-dir mc_dropout_uncertainty
"""

import argparse
import csv
import glob
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).parent
_EPS  = 1e-12

# Keys written to every output npz / csv (in order)
_OUT_KEYS = [
    'time', 'action_idx',
    'mc_total_entropy', 'mc_within_entropy', 'mc_across_entropy',
    'mc_predictive_variance', 'mc_vote_entropy', 'mc_agreement',
    'mc_human_action_prob', 'mc_human_action_surprisal',
    'mc_confidence', 'mc_margin',
    'mc_entropy_delta', 'mc_entropy_increase', 'mc_entropy_decrease',
]


# ── Utilities ──────────────────────────────────────────────────────────────────

def safe_save(obj, path):
    """Atomic save: write to .tmp then rename to prevent partial-write corruption."""
    path = Path(path)
    tmp  = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_root_to_path():
    """Ensure project root is on sys.path for model imports."""
    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='MC Dropout uncertainty extractor for BC (Behavior Cloning)'
    )
    p.add_argument('--mode', choices=['train', 'extract'], default='train',
                   help='train: train BCDropout; extract: run MC inference')

    # Data
    p.add_argument('--data-dir', default=str(_ROOT / 'processed_data_frameskip_4'),
                   help='Base directory containing processed subject data')
    p.add_argument('--subject', default='sub_1',
                   choices=['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6'])
    p.add_argument('--file-idx', nargs='+', type=int, default=list(range(11)),
                   help='File indices to process '
                        '(train mode: training files; extract mode: files to run)')
    p.add_argument('--val-file-idx', type=int, default=10,
                   help='(train mode) File index for validation (excluded from train)')
    p.add_argument('--batch-size', type=int, default=256,
                   help='Mini-batch size for training and MC inference')

    # Model
    p.add_argument('--dqn-path', default=str(_ROOT / 'pretrained' / 'dqn_cnn.pt'),
                   help='Path to pretrained DQN CNN checkpoint')
    p.add_argument('--bc-path', default='',
                   help='(extract mode) Path to trained BCDropout checkpoint')
    p.add_argument('--dropout-p', type=float, default=0.1,
                   help='Dropout probability inside the MLP head')
    p.add_argument('--mc-samples', type=int, default=30,
                   help='Number of stochastic forward passes for MC Dropout inference')
    p.add_argument('--use-mc-dropout', action='store_true', default=True,
                   help='Enable MC Dropout at inference (default: True)')

    # Training
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save-dir', default=str(_ROOT / 'checkpoints'),
                   help='Root save dir; checkpoints go to save_dir/bc_dropout/<subject>/')
    p.add_argument('--save-interval', type=int, default=1,
                   help='Checkpoint save interval (epochs)')

    # Output (extract mode)
    p.add_argument('--out-dir', default=str(_ROOT / 'mc_dropout_uncertainty'),
                   help='Directory to save per-run uncertainty npz/csv files')

    # Misc
    p.add_argument('--seed',   type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    return p.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────────

class BCDropout(nn.Module):
    """
    Behavior Cloning model with Dropout in the MLP head.

    Architecture:
        DQN_CNN (frozen, always eval)
            -> Linear(3136, 512) -> ReLU -> Dropout(p) -> Linear(512, action_dim)

    The CNN is kept in eval mode unconditionally (no BatchNorm; conv layers
    are deterministic in train/eval). Dropout is live during training and
    selectively re-enabled at inference via enable_dropout_only().

    Input to forward(): (B, 4, 84, 84) float32 in [0, 1].
    """

    def __init__(self, cnn, action_dim=6, dropout_p=0.1):
        super().__init__()
        self.cnn        = cnn
        self.action_dim = action_dim
        self.dropout_p  = dropout_p
        # CNN is always frozen
        for param in self.cnn.parameters():
            param.requires_grad_(False)

        self.action_head = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, action_dim),
        )

    def train(self, mode=True):
        """Keep CNN in eval mode regardless of the overall train/eval state."""
        super().train(mode)
        self.cnn.eval()
        return self

    def forward(self, state):
        """
        Args:
            state: (B, 4, 84, 84) float32 [0, 1]
        Returns:
            logits: (B, action_dim)
        """
        with torch.no_grad():
            feat = self.cnn(state)          # (B, 3136); no grad needed (frozen CNN)
        return self.action_head(feat)       # (B, A); grad flows here during training

    def get_action(self, state, deterministic=True):
        """Compatible with eval.evaluate_agent interface."""
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(np.asarray(state)).float()
        if state.dim() == 3:
            state = state.unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(state)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
        return action.item() if action.numel() == 1 else action


# ── MC Dropout activation ─────────────────────────────────────────────────────

def enable_dropout_only(model):
    """
    Put model in eval mode, then re-enable all nn.Dropout layers.

    This keeps BatchNorm (if any) and other stochastic layers deterministic,
    while allowing Dropout to generate different masks per forward pass.
    Call this before every MC inference loop, not just once at load time.

    Returns:
        model (mutated in-place, also returned for chaining)
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    return model


# ── Public API ────────────────────────────────────────────────────────────────

def load_bc_mc_model(path, dropout_p, device):
    """
    Load a trained BCDropout model from a checkpoint saved by train_mode().

    The checkpoint must contain a 'model_state' key (state_dict) or be a
    raw state_dict.  dropout_p must match the training configuration.

    Args:
        path      : str or Path — .pth checkpoint
        dropout_p : float — must match the model that was trained
        device    : str — 'cpu' or 'cuda'
    Returns:
        BCDropout in eval mode on device
    """
    _add_root_to_path()
    from model.dqn import DQN_CNN

    cnn   = DQN_CNN()
    model = BCDropout(cnn, action_dim=6, dropout_p=dropout_p).to(device)

    ckpt       = torch.load(str(path), map_location=device)
    state_dict = ckpt.get('model_state', ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'  BCDropout loaded: {path}', flush=True)
    return model


def extract_mc_dropout_uncertainty(model, states_uint8, actions,
                                    mc_samples, batch_size, device):
    """
    Run MC Dropout inference and return all uncertainty metrics.

    Integration point for GLM regressor scripts — pass the loaded model and
    raw uint8 states from any run's npz file.

    Args:
        model       : BCDropout, loaded via load_bc_mc_model()
        states_uint8: (N, 4, 84, 84) uint8 numpy array
        actions     : (N,) int64 numpy array — human action class indices
        mc_samples  : int — number of stochastic forward passes
        batch_size  : int — inference mini-batch size
        device      : str
    Returns:
        dict with the following (N,) float32 arrays
        (plus 'action_idx' as int32):
          action_idx, mc_total_entropy, mc_within_entropy, mc_across_entropy,
          mc_predictive_variance, mc_vote_entropy, mc_agreement,
          mc_human_action_prob, mc_human_action_surprisal, mc_confidence,
          mc_margin, mc_entropy_delta, mc_entropy_increase, mc_entropy_decrease
    """
    mean_probs, probs_mc = _mc_forward(model, states_uint8, mc_samples, batch_size, device)
    result = _compute_uncertainty(mean_probs, probs_mc, actions)
    result['action_idx'] = actions.astype(np.int32)
    return result


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_run(npz_path):
    """
    Load a single run .npz file.

    Returns:
        states  : (N, 4, 84, 84) uint8
        actions : (N,) int64   — argmax of one-hot if stored as (N, 6)
        time    : (N,) float32
    """
    data    = np.load(str(npz_path))
    states  = data['state']                 # (N, 4, 84, 84) uint8
    actions = data['action']
    time    = (data['time'].astype(np.float32)
               if 'time' in data
               else np.arange(len(states), dtype=np.float32))

    if actions.ndim == 2:                   # (N, 6) one-hot → index
        actions = actions.argmax(axis=1).astype(np.int64)
    else:
        actions = actions.astype(np.int64)

    return states, actions, time


def _load_runs(npz_files):
    """Concatenate multiple runs into single arrays."""
    s_list, a_list, t_list = [], [], []
    for f in npz_files:
        s, a, t = _load_run(f)
        s_list.append(s); a_list.append(a); t_list.append(t)
    return (np.concatenate(s_list),
            np.concatenate(a_list),
            np.concatenate(t_list))


def _resolve_files(data_dir, subject, file_idx):
    """Return sorted list of (index, path) tuples for given file_idx."""
    subject_dir = Path(data_dir) / subject
    npz_files   = sorted(glob.glob(str(subject_dir / '*.npz')))
    if not npz_files:
        raise FileNotFoundError(f'No .npz files in {subject_dir}')
    result = []
    for idx in file_idx:
        if idx < 0 or idx >= len(npz_files):
            print(f'  WARNING: file_idx {idx} out of range (n={len(npz_files)}), skip',
                  flush=True)
            continue
        result.append((idx, npz_files[idx]))
    return result


def _load_cnn(dqn_path, device):
    """Load pretrained DQN_CNN encoder, always frozen."""
    _add_root_to_path()
    from model.dqn import load_pretrained_cnn
    cnn = load_pretrained_cnn(str(dqn_path), freeze=True)
    cnn.to(device).eval()
    return cnn


# ── Training helpers ──────────────────────────────────────────────────────────

def _run_epoch(model, states, actions, batch_size, device, optimizer=None):
    """
    One full pass over (states, actions).

    Training when optimizer is provided; validation otherwise.
    Returns (mean_CE, accuracy, mean_H_pi) where mean_H_pi is the mean
    policy entropy of the *mean* logits (MC Dropout disabled, deterministic).

    Args:
        model    : BCDropout
        states   : (N, 4, 84, 84) uint8 numpy
        actions  : (N,) int64 numpy
        batch_size: int
        device   : str
        optimizer: torch.optim.Optimizer or None
    Returns:
        mean_ce  : float
        accuracy : float
        mean_H   : float
    """
    is_train = optimizer is not None
    if is_train:
        model.train()       # CNN stays eval (overridden); Dropout active
    else:
        model.eval()        # Dropout off for deterministic val

    N = len(states)
    perm = np.random.permutation(N) if is_train else np.arange(N)

    total_ce, total_correct, total_H = 0.0, 0, 0.0
    n_batches = 0

    for start in range(0, N, batch_size):
        bidx = perm[start:start + batch_size]
        x = torch.from_numpy(states[bidx].astype(np.float32) / 255.0).to(device)
        a = torch.from_numpy(actions[bidx]).long().to(device)

        logits = model(x)                                       # (B, A)
        ce     = F.cross_entropy(logits, a)

        if is_train:
            optimizer.zero_grad()
            ce.backward()
            optimizer.step()

        with torch.no_grad():
            probs   = F.softmax(logits.detach(), dim=-1)        # (B, A)
            H       = -(probs * (probs + _EPS).log()).sum(dim=-1).mean().item()
            correct = (logits.detach().argmax(dim=-1) == a).sum().item()

        total_ce      += ce.item()
        total_correct += correct
        total_H       += H
        n_batches     += 1

    return (total_ce / max(n_batches, 1),
            total_correct / max(N, 1),
            total_H / max(n_batches, 1))


# ── MC Dropout inference ──────────────────────────────────────────────────────

def _mc_forward(model, states_uint8, mc_samples, batch_size, device):
    """
    Run T stochastic forward passes with MC Dropout.

    Processes one sample t at a time in mini-batches to avoid stacking a full
    (T, N, A) tensor on the GPU.  Each sample is accumulated as CPU numpy.

    Args:
        model       : BCDropout
        states_uint8: (N, 4, 84, 84) uint8 numpy
        mc_samples  : int T
        batch_size  : int
        device      : str
    Returns:
        mean_probs: (N, A) float32 numpy
        probs_mc  : (T, N, A) float32 numpy — all on CPU
    """
    enable_dropout_only(model)   # eval + re-enable Dropout

    N        = len(states_uint8)
    all_probs = []               # list of (N, A) per MC sample

    for _t in range(mc_samples):
        probs_t = []
        for start in range(0, N, batch_size):
            x = torch.from_numpy(
                states_uint8[start:start + batch_size].astype(np.float32) / 255.0
            ).to(device)
            with torch.no_grad():
                p = F.softmax(model(x), dim=-1).cpu().numpy()  # (b, A)
            probs_t.append(p)
        all_probs.append(np.concatenate(probs_t, axis=0))      # (N, A)

    probs_mc   = np.stack(all_probs, axis=0).astype(np.float32)   # (T, N, A)
    mean_probs = probs_mc.mean(axis=0)                             # (N, A)
    return mean_probs, probs_mc


# ── Uncertainty computation ───────────────────────────────────────────────────

def _compute_uncertainty(mean_probs, probs_mc, actions):
    """
    Compute all frame-level uncertainty metrics from MC Dropout samples.

    Naming convention (avoids over-claiming causal interpretation):
      total entropy   = H(mean_probs)           predictive entropy
      within entropy  = mean_t H(probs_t)       aleatoric-like
      across entropy  = H(mean) - mean_t H      MC-dropout disagreement
                                                 clipped to >= 0 to remove float error
      predictive variance = mean_a Var_t[p_t(a|s)]
      vote entropy    = H of argmax-vote distribution over T samples
      agreement       = max vote proportion
      human prob      = mean_probs[a_human]
      human surprisal = -log(human_prob + eps)
      confidence      = max_a mean_probs[a]
      margin          = top1 - top2 prob
      entropy delta   = H_total[t] - H_total[t-1]  (0 at first frame)
      entropy increase/decrease = ReLU(±delta)

    Args:
        mean_probs: (N, A) float32
        probs_mc  : (T, N, A) float32
        actions   : (N,) int64 — human action indices
    Returns:
        dict of (N,) float32 numpy arrays
    """
    eps        = _EPS
    T, N, A    = probs_mc.shape

    # Total entropy H(mean_probs)
    total_ent  = -(mean_probs * np.log(mean_probs + eps)).sum(axis=-1)   # (N,)

    # Within entropy = mean_t H(probs_t)
    per_t_ent  = -(probs_mc * np.log(probs_mc + eps)).sum(axis=-1)       # (T, N)
    within_ent = per_t_ent.mean(axis=0)                                   # (N,)

    # Across entropy = H(mean) - mean H  — clip float-error negatives
    across_ent = np.maximum(total_ent - within_ent, 0.0)                 # (N,)

    # Predictive variance: mean over action dim of per-action sample variance
    pred_var   = probs_mc.var(axis=0).mean(axis=-1)                      # (N,)

    # Vote entropy and agreement
    votes      = probs_mc.argmax(axis=-1)                                 # (T, N)
    vote_counts = np.zeros((N, A), dtype=np.float32)
    for t in range(T):
        vote_counts[np.arange(N), votes[t]] += 1.0
    vote_p    = vote_counts / T                                           # (N, A)
    vote_ent  = -(vote_p * np.log(vote_p + eps)).sum(axis=-1)            # (N,)
    agreement = vote_p.max(axis=-1)                                       # (N,)

    # Human action probability and surprisal
    human_prob      = mean_probs[np.arange(N), actions]                  # (N,)
    human_surprisal = -np.log(human_prob + eps)                          # (N,)

    # Confidence (max prob) and margin (top1 - top2)
    sorted_p   = np.sort(mean_probs, axis=-1)[:, ::-1]
    confidence = sorted_p[:, 0].copy()                                   # (N,)
    margin     = (sorted_p[:, 0] - sorted_p[:, 1]).copy()               # (N,)

    # Entropy delta, increase, decrease
    entropy_delta    = np.zeros(N, dtype=np.float32)
    entropy_delta[1:] = (total_ent[1:] - total_ent[:-1]).astype(np.float32)
    entropy_increase = np.maximum( entropy_delta, 0.0)
    entropy_decrease = np.maximum(-entropy_delta, 0.0)

    return dict(
        mc_total_entropy          = total_ent.astype(np.float32),
        mc_within_entropy         = within_ent.astype(np.float32),
        mc_across_entropy         = across_ent.astype(np.float32),
        mc_predictive_variance    = pred_var.astype(np.float32),
        mc_vote_entropy           = vote_ent.astype(np.float32),
        mc_agreement              = agreement.astype(np.float32),
        mc_human_action_prob      = human_prob.astype(np.float32),
        mc_human_action_surprisal = human_surprisal.astype(np.float32),
        mc_confidence             = confidence.astype(np.float32),
        mc_margin                 = margin.astype(np.float32),
        mc_entropy_delta          = entropy_delta,
        mc_entropy_increase       = entropy_increase,
        mc_entropy_decrease       = entropy_decrease,
    )


# ── Diagnostics ───────────────────────────────────────────────────────────────

def _print_diagnostics(unc, actions, mean_probs):
    """Print per-run uncertainty summary statistics."""
    N      = len(unc['mc_total_entropy'])
    A      = mean_probs.shape[-1]
    max_H  = np.log(A)
    acc    = (mean_probs.argmax(axis=-1) == actions).mean()

    def _stats(arr, name):
        q = np.quantile(arr, [0.5, 0.9, 0.95])
        print(
            f'    {name:<34s}: mean={arr.mean():.4f}  std={arr.std():.4f}'
            f'  p50={q[0]:.4f}  p90={q[1]:.4f}  p95={q[2]:.4f}',
            flush=True,
        )

    print(f'  [Uncertainty diagnostics | N={N:,} | max H={max_H:.4f}]', flush=True)
    _stats(unc['mc_total_entropy'],          'total entropy')
    _stats(unc['mc_within_entropy'],         'within entropy')
    _stats(unc['mc_across_entropy'],         'across entropy (MC disagree)')
    _stats(unc['mc_predictive_variance'],    'predictive variance')
    _stats(unc['mc_vote_entropy'],           'vote entropy')
    _stats(unc['mc_agreement'],              'agreement')
    print(f'    {"human action prob mean":<34s}: {unc["mc_human_action_prob"].mean():.4f}',
          flush=True)
    print(f'    {"human action surprisal mean":<34s}: {unc["mc_human_action_surprisal"].mean():.4f}',
          flush=True)
    print(f'    {"accuracy (mean_probs argmax)":<34s}: {acc:.4f}', flush=True)


# ── Save ──────────────────────────────────────────────────────────────────────

def _save_results(out_dir, subject, fidx, data_dict):
    """
    Save per-run uncertainty dict as .npz (compressed) and .csv.

    Only keys present in _OUT_KEYS and data_dict are written, in _OUT_KEYS order.
    """
    out_path = Path(out_dir) / subject
    out_path.mkdir(parents=True, exist_ok=True)

    stem     = f'mc_dropout_uncertainty_{subject}_f{fidx:02d}'
    npz_path = out_path / f'{stem}.npz'
    csv_path = out_path / f'{stem}.csv'

    present  = [k for k in _OUT_KEYS if k in data_dict]

    # npz
    np.savez_compressed(str(npz_path), **{k: data_dict[k] for k in present})
    print(f'  Saved npz : {npz_path}', flush=True)

    # csv
    N = len(next(iter(data_dict.values())))
    with open(str(csv_path), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=present)
        writer.writeheader()
        for i in range(N):
            writer.writerow({k: float(data_dict[k][i]) for k in present})
    print(f'  Saved csv : {csv_path}', flush=True)


# ── Train mode ────────────────────────────────────────────────────────────────

def train_mode(args):
    """Train a BCDropout model with CE loss on human trajectory data."""
    set_seed(args.seed)
    device = args.device

    # ── Resolve files ──────────────────────────────────────────────────────────
    subject_dir = Path(args.data_dir) / args.subject
    npz_files   = sorted(glob.glob(str(subject_dir / '*.npz')))
    if not npz_files:
        raise FileNotFoundError(f'No .npz files in {subject_dir}')

    train_idx = [i for i in args.file_idx if i != args.val_file_idx]
    val_idx   = args.val_file_idx

    print(f'\n[Data] {args.subject}')
    print(f'  Train files : {train_idx}')
    print(f'  Val file    : {val_idx}')

    train_files = [npz_files[i] for i in train_idx if i < len(npz_files)]
    val_files   = ([npz_files[val_idx]] if val_idx < len(npz_files) else [])

    print('  Loading train data ...')
    train_s, train_a, _ = _load_runs(train_files)
    print(f'    {len(train_s):,} frames', flush=True)

    val_s = val_a = None
    if val_files:
        print('  Loading val data ...')
        val_s, val_a, _ = _load_runs(val_files)
        print(f'    {len(val_s):,} frames', flush=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f'\n[Model] dropout_p={args.dropout_p}  dqn={args.dqn_path}')
    cnn   = _load_cnn(args.dqn_path, device)
    model = BCDropout(cnn, action_dim=6, dropout_p=args.dropout_p).to(device)
    opt   = torch.optim.Adam(model.action_head.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.action_head.parameters())
    print(f'  Trainable params (head only): {n_params:,}')

    save_dir = Path(args.save_dir) / 'bc_dropout' / args.subject
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'  Checkpoints -> {save_dir}')
    print('=' * 80)

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        tr_ce, tr_acc, _ = _run_epoch(
            model, train_s, train_a, args.batch_size, device, opt
        )

        val_ce = val_acc = val_H = float('nan')
        if val_s is not None:
            val_ce, val_acc, val_H = _run_epoch(
                model, val_s, val_a, args.batch_size, device
            )

        print(
            f'Epoch {epoch:3d}/{args.epochs}  '
            f'train_CE={tr_ce:.4f}  train_acc={tr_acc:.4f}  '
            f'val_CE={val_ce:.4f}  val_acc={val_acc:.4f}  '
            f'val_H_π={val_H:.4f}',
            flush=True,
        )

        if epoch % args.save_interval == 0:
            ckpt = {
                'model_state': model.state_dict(),
                'epoch'      : epoch,
                'dropout_p'  : args.dropout_p,
                'args'       : vars(args),
            }
            safe_save(ckpt, save_dir / f'epoch_{epoch:03d}.pth')
            print(f'  Saved -> {save_dir}/epoch_{epoch:03d}.pth', flush=True)

    # Final checkpoint
    ckpt = {
        'model_state': model.state_dict(),
        'epoch'      : args.epochs,
        'dropout_p'  : args.dropout_p,
        'args'       : vars(args),
    }
    safe_save(ckpt, save_dir / 'final.pth')
    print(f'\n[Done] Final model -> {save_dir}/final.pth')


# ── Extract mode ──────────────────────────────────────────────────────────────

def extract_mode(args):
    """Run MC Dropout inference over specified runs and save uncertainty regressors."""
    set_seed(args.seed)
    device = args.device

    if not args.bc_path:
        raise ValueError('--bc-path must be specified in extract mode')

    print(f'\n[Extract] subject={args.subject}  T={args.mc_samples}')
    print(f'  Model    : {args.bc_path}')
    print(f'  Dropout p: {args.dropout_p}')

    model = load_bc_mc_model(args.bc_path, args.dropout_p, device)
    runs  = _resolve_files(args.data_dir, args.subject, args.file_idx)

    for fidx, fpath in runs:
        print(f'\n{"="*80}', flush=True)
        print(f'  Run f{fidx:02d}: {Path(fpath).name}', flush=True)

        states, actions, time = _load_run(fpath)
        print(f'  Frames: {len(states):,}', flush=True)
        print(f'  Running {args.mc_samples} MC forward passes ...', flush=True)

        mean_probs, probs_mc = _mc_forward(
            model, states, args.mc_samples, args.batch_size, device
        )

        unc = _compute_uncertainty(mean_probs, probs_mc, actions)
        _print_diagnostics(unc, actions, mean_probs)

        result = {'time': time, 'action_idx': actions.astype(np.int32)}
        result.update(unc)
        _save_results(args.out_dir, args.subject, fidx, result)

    print(f'\n[Done] Results saved to {args.out_dir}/{args.subject}/')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = get_args()

    print('=' * 80)
    print('MC Dropout BC Uncertainty Extractor')
    print(f'  Mode      : {args.mode}')
    print(f'  Subject   : {args.subject}')
    print(f'  Data dir  : {args.data_dir}')
    print(f'  Dropout p : {args.dropout_p}')
    print(f'  MC samples: {args.mc_samples}')
    print(f'  Device    : {args.device}')
    print('=' * 80)

    if args.mode == 'train':
        train_mode(args)
    else:
        extract_mode(args)


if __name__ == '__main__':
    main()
