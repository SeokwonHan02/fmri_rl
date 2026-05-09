"""
behavioral.py

Ensemble DQN uncertainty analysis for a single subject across multiple runs.
For each entry in RUNS = [(run_idx, model_fname), ...]:
  - loads the model trained with run_idx held out
  - applies it to run_idx data → per-frame PEN uncertainty
  - computes action accuracy (model prediction vs human) and future switch rate

All runs are pooled for analysis and visualisation.

Outputs (OUT_DIR/):
  uncertainty_dist.png        — aleatoric / epistemic / total (FIRE vs non-FIRE)
  action_accuracy_binned.png  — uncertainty vs action accuracy (binned mean ± SEM)
  action_switch_binned.png    — uncertainty vs future switch rate (binned mean ± SEM)
  scatter_alea_epi.png        — aleatoric vs epistemic scatter (colour = switch rate)
  corr_bar.png                — Pearson r of each uncertainty with accuracy / switch rate
  summary_stats.csv
"""

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import softmax as _softmax

_HERE = Path(__file__).resolve().parent   # ensemble/
_ROOT = _HERE.parent                       # fMRI_RL/

# ── Configuration ─────────────────────────────────────────────────────────────
SUBJECT   = 'sub_1'
DATA_DIR  = _ROOT / 'processed_data_frameskip_4' / 'sub_1'
MODEL_DIR = _HERE / 'trained_ensemble_model'
DQN_PATH  = _ROOT / 'pretrained' / 'dqn_cnn.pt'
OUT_DIR   = _HERE / 'results'

# (run_index_in_sorted_npz_list, model_filename)
# Each model was trained leaving out the corresponding run_index.
RUNS = [
    (0,  'sub_1_run_0.pth'),
    (1,  'sub_1_run_1.pth'),
    (10, 'sub_1_run_10.pth'),
]

HORIZON_SEC = 1.0
N_BINS      = 10
EPS         = 1e-12
BATCH_SIZE  = 512
AGG            = 'mean'   # Q aggregation across heads: 'mean' or 'min'
PRE_POST_FRAMES = 15      # frames before/after each NOOP run for Q.C
KNN_K           = 50       # k for kNN state-rarity computation

# Direction group: NOOP/FIRE→0, RIGHT/RIGHT+FIRE→1, LEFT/LEFT+FIRE→2
_ACTION_TO_DIR = np.array([0, 0, 1, 2, 1, 2], dtype=np.int32)

# ── DQN model ────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DQN(nn.Module):
    def __init__(self, n_actions: int = 6):
        super().__init__()
        self.conv1  = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2  = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3  = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc3    = nn.Linear(3136, 512)
        self.fc_out = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc3(h))
        return self.fc_out(h)


# ── Model loading ─────────────────────────────────────────────────────────────

def _robust_load(path, map_location='cpu'):
    p = str(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(p, map_location=map_location)


def load_dqn(path: Path) -> _DQN:
    ckpt = _robust_load(path)
    sd   = ckpt.get('policy_net', ckpt)
    dqn  = _DQN()
    dqn.load_state_dict(sd)
    dqn.eval()
    print(f'  DQN loaded: {path}')
    return dqn


def load_ensemble_heads(model_path: Path, dqn_model: _DQN) -> nn.ModuleList:
    heads_sd = _robust_load(model_path)
    head_indices: set[int] = set()
    for k in heads_sd:
        try:
            head_indices.add(int(k.split('.')[0]))
        except ValueError:
            pass
    if not head_indices:
        raise ValueError(
            f'Cannot parse head indices from {model_path}. '
            f'Sample keys: {list(heads_sd.keys())[:6]}'
        )
    num_heads = max(head_indices) + 1
    heads = nn.ModuleList([
        nn.Sequential(nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, 6))
        for _ in range(num_heads)
    ])
    heads.load_state_dict(heads_sd)
    heads.eval()
    print(f'  Ensemble loaded: {Path(model_path).name}  ({num_heads} heads)')
    return heads


# ── Inference ─────────────────────────────────────────────────────────────────

def extract_q_all(heads: nn.ModuleList,
                  dqn_model: _DQN,
                  states_uint8: np.ndarray) -> np.ndarray:
    """(N, 4, 84, 84) uint8 → (N, H, 6) float64"""
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states_uint8), BATCH_SIZE):
            x = torch.from_numpy(
                states_uint8[start:start + BATCH_SIZE]
            ).float() / 255.0
            h = F.relu(dqn_model.conv1(x))
            h = F.relu(dqn_model.conv2(h))
            h = F.relu(dqn_model.conv3(h))
            feat  = h.view(h.size(0), -1)
            q_all = torch.stack([hd(feat) for hd in heads], dim=1)
            chunks.append(q_all.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


# ── Uncertainty decomposition ─────────────────────────────────────────────────

def compute_uncertainty(q_all: np.ndarray,
                        temp: float = 1.0,
                        agg: str = 'mean') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PEN decomposition with mean-centering and temperature T.
    agg='mean': consensus = mean of per-head softmaxes (mixture).
    agg='min':  consensus = softmax(min_h Q / T).
    Returns aleatoric, epistemic, total — each (N,) float64.
    """
    zq_all    = q_all - q_all.mean(axis=2, keepdims=True)         # (N, H, 6)
    probs_all = _softmax(zq_all / temp, axis=2)                    # (N, H, 6)

    if agg == 'mean':
        probs_agg = probs_all.mean(axis=1)                         # (N, 6) mixture
    else:
        q_min     = q_all.min(axis=1)                              # (N, 6)
        zq_min    = q_min - q_min.mean(axis=1, keepdims=True)
        probs_agg = _softmax(zq_min / temp, axis=1)                # (N, 6)

    total     = -(probs_agg * np.log(probs_agg + EPS)).sum(axis=1)
    aleatoric = -(probs_all * np.log(probs_all + EPS)).sum(axis=2).mean(axis=1)
    epistemic = total - aleatoric
    return aleatoric.astype(np.float64), epistemic.astype(np.float64), total.astype(np.float64)


def calibrate_temperature(heads: nn.ModuleList, dqn_model: _DQN,
                          npz_files: list,
                          agg: str = 'mean') -> tuple[float, float, float]:
    """
    Find T* minimising NLL of human actions under softmax(zQ_agg / T).
    Optimises in log(T) space, bounds (-4, 4).
    Returns T_opt, nll_at_T1, nll_at_Topt.
    """
    from scipy.optimize import minimize_scalar

    zq_list, act_list = [], []
    for npz_path in npz_files:
        data          = np.load(npz_path)
        states_np     = data['state']
        acts          = data['action']
        actual_action = acts.argmax(axis=1) if acts.ndim == 2 else acts.astype(int)

        chunks = []
        with torch.no_grad():
            for start in range(0, len(states_np), BATCH_SIZE):
                x = torch.from_numpy(
                    states_np[start:start + BATCH_SIZE]
                ).float() / 255.0
                h = F.relu(dqn_model.conv1(x))
                h = F.relu(dqn_model.conv2(h))
                h = F.relu(dqn_model.conv3(h))
                feat  = h.view(h.size(0), -1)
                q_all = torch.stack([hd(feat) for hd in heads], dim=1)
                chunks.append(q_all.cpu().numpy())

        q_all = np.concatenate(chunks, axis=0).astype(np.float64)
        q_agg = q_all.mean(axis=1) if agg == 'mean' else q_all.min(axis=1)
        zq_agg = q_agg - q_agg.mean(axis=1, keepdims=True)
        zq_list.append(zq_agg)
        act_list.append(actual_action)

    zq_agg_all = np.concatenate(zq_list, axis=0)
    actions     = np.concatenate(act_list)

    def nll(log_T):
        T     = np.exp(log_T)
        probs = _softmax(zq_agg_all / T, axis=1)
        return -np.log(probs[np.arange(len(actions)), actions] + EPS).mean()

    result  = minimize_scalar(nll, bounds=(-4.0, 4.0), method='bounded')
    T_opt   = float(np.exp(result.x))
    nll_t1  = float(nll(0.0))
    nll_opt = float(result.fun)
    return T_opt, nll_t1, nll_opt


# ── Behavioral helpers ────────────────────────────────────────────────────────

def compute_switch_indicator(actual_action: np.ndarray) -> np.ndarray:
    """switch[j] = 1 if direction group changed from j-1 to j, else 0."""
    direction = _ACTION_TO_DIR[actual_action]
    sw = np.zeros(len(direction), dtype=np.float64)
    sw[1:] = (direction[1:] != direction[:-1]).astype(np.float64)
    return sw


def future_window_mean(time: np.ndarray, values: np.ndarray,
                       horizon: float) -> np.ndarray:
    """Mean of values in (time[i], time[i]+horizon] for each i. NaN if empty."""
    N       = len(time)
    cumval  = np.concatenate([[0.0], np.cumsum(values)])
    j_end   = np.searchsorted(time, time + horizon, side='right')
    j_start = np.arange(N) + 1
    counts  = (j_end - j_start).astype(np.float64)
    total   = cumval[j_end] - cumval[j_start]
    return np.where(counts > 0, total / counts, np.nan).astype(np.float64)


# ── Statistics helpers ────────────────────────────────────────────────────────

def binned_stats(x: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Percentile-binned mean ± SEM, ignoring NaN."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    edges      = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    edges[-1] += 1e-9
    centers, means, sems, ns = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m    = (x >= lo) & (x < hi)
        vals = y[m]
        centers.append(np.median(x[m]) if m.any() else (lo + hi) / 2)
        means.append(vals.mean()        if len(vals)     else np.nan)
        sems.append(stats.sem(vals)     if len(vals) > 1 else np.nan)
        ns.append(m.sum())
    return (np.array(centers), np.array(means),
            np.array(sems),    np.array(ns, dtype=int))


def pearsonr_ci(x: np.ndarray, y: np.ndarray,
                n_boot: int = 1000, ci: float = 0.95,
                rng: np.random.Generator | None = None):
    """Pearson r with bootstrap CI. Returns r, p, (lo, hi)."""
    r, p = stats.pearsonr(x, y)
    if rng is None:
        rng = np.random.default_rng(42)
    n  = len(x)
    rs = np.array([
        stats.pearsonr(x[idx := rng.integers(0, n, n)], y[idx])[0]
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.nanpercentile(rs, [100 * alpha, 100 * (1 - alpha)])
    return r, p, (lo, hi)


# ── Plotting ──────────────────────────────────────────────────────────────────

UNCERTAINTY_CFG = [
    ('aleatoric', '#1f77b4', 'Aleatoric'),
    ('epistemic', '#d62728', 'Epistemic'),
    ('total',     '#ff7f0e', 'Total entropy'),
]


def plot_uncertainty_dist(unc_dict: dict, actual_action: np.ndarray,
                          out_path: Path):
    """Aleatoric / epistemic / total distributions split by FIRE vs non-FIRE."""
    is_fire   = (actual_action == 1) | (actual_action == 4) | (actual_action == 5)
    is_nofire = ~is_fire

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, color, label) in zip(axes, UNCERTAINTY_CFG):
        arr = unc_dict[key]
        ax.hist(arr[is_fire],   bins=60, alpha=0.55, color='#d62728',
                density=True, label=f'FIRE (n={is_fire.sum():,})', linewidth=0)
        ax.hist(arr[is_nofire], bins=60, alpha=0.45, color='#1f77b4',
                density=True, label=f'non-FIRE (n={is_nofire.sum():,})', linewidth=0)
        ax.axvline(np.log(6), color='black', lw=1, ls='--', label='max=ln(6)')
        ks, ks_p = stats.ks_2samp(arr[is_fire], arr[is_nofire])
        ax.set_title(f'{label}\nKS={ks:.3f}, p={ks_p:.2e}', fontsize=9)
        ax.set_xlabel('Uncertainty', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{SUBJECT} — Uncertainty Distribution (FIRE vs non-FIRE)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def plot_action_accuracy_binned(unc_dict: dict, correct: np.ndarray,
                                out_path: Path):
    """3-panel: each uncertainty type vs action accuracy (binned mean ± SEM)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, color, label) in zip(axes, UNCERTAINTY_CFG):
        x = unc_dict[key]
        centers, means, sems, _ = binned_stats(x, correct, N_BINS)
        r, p = stats.pearsonr(x, correct)

        ax.fill_between(centers, means - sems, means + sems,
                        alpha=0.25, color=color)
        ax.plot(centers, means, 'o-', color=color, lw=2, ms=5)
        ax.axvline(np.log(6), color='gray', lw=0.8, ls='--', alpha=0.5,
                   label='max=ln(6)')
        ax.axhline(1 / 6, color='black', lw=0.8, ls=':', alpha=0.5,
                   label='chance (1/6)')
        ax.set_title(f'{label}\nr={r:.3f}, p={p:.2e}', fontsize=9)
        ax.set_xlabel('Uncertainty', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Action accuracy (model = human)', fontsize=9)
    fig.suptitle(
        f'{SUBJECT} — Uncertainty vs Action Accuracy  '
        f'(N={len(correct):,}, overall={correct.mean():.3f})',
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def plot_action_switch_binned(unc_dict: dict, future_sw: np.ndarray,
                              out_path: Path):
    """3-panel: each uncertainty type vs future action switch rate (binned)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    valid = np.isfinite(future_sw)

    for ax, (key, color, label) in zip(axes, UNCERTAINTY_CFG):
        x = unc_dict[key][valid]
        y = future_sw[valid]
        centers, means, sems, _ = binned_stats(x, y, N_BINS)
        r, p = stats.pearsonr(x, y)

        ax.fill_between(centers, means - sems, means + sems,
                        alpha=0.25, color=color)
        ax.plot(centers, means, 'o-', color=color, lw=2, ms=5)
        ax.axvline(np.log(6), color='gray', lw=0.8, ls='--', alpha=0.5,
                   label='max=ln(6)')
        ax.set_title(f'{label}\nr={r:.3f}, p={p:.2e}', fontsize=9)
        ax.set_xlabel('Uncertainty', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f'Future action switch rate\n(next {HORIZON_SEC:.0f}s)', fontsize=9)
    fig.suptitle(
        f'{SUBJECT} — Uncertainty vs Future Action Switch Rate  (N={valid.sum():,})',
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def plot_scatter_alea_epi(unc_dict: dict, future_sw: np.ndarray,
                          out_path: Path):
    """Aleatoric vs epistemic scatter, colour = future switch rate."""
    valid = np.isfinite(future_sw)
    alea  = unc_dict['aleatoric'][valid]
    epi   = unc_dict['epistemic'][valid]
    sw    = future_sw[valid]

    N = len(alea)
    if N > 30_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, 30_000, replace=False)
        alea, epi, sw = alea[idx], epi[idx], sw[idx]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(alea, epi, c=sw, cmap='RdYlBu_r',
                    s=3, alpha=0.45, vmin=0, vmax=sw.max(), linewidths=0)
    plt.colorbar(sc, ax=ax, label=f'Future switch rate (next {HORIZON_SEC:.0f}s)')
    ax.set_xlabel('Aleatoric uncertainty', fontsize=10)
    ax.set_ylabel('Epistemic uncertainty', fontsize=10)
    ax.set_title(
        f'{SUBJECT} — Aleatoric vs Epistemic  (shown {len(alea):,} pts)',
        fontsize=10
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def plot_corr_bar(unc_dict: dict, correct: np.ndarray,
                  future_sw: np.ndarray, out_path: Path):
    """2-panel Pearson r bar: uncertainty vs action accuracy & switch rate."""
    rng     = np.random.default_rng(42)
    keys    = [k for k, *_ in UNCERTAINTY_CFG]
    cols    = [c for _, c, _ in UNCERTAINTY_CFG]
    lbls    = [l for _, _, l in UNCERTAINTY_CFG]
    x_pos   = np.arange(len(keys))
    valid_s = np.isfinite(future_sw)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (arr, mask, title) in zip(axes, [
        (correct,   np.ones(len(correct), dtype=bool),
         'Action Accuracy'),
        (future_sw, valid_s,
         f'Future Switch Rate (next {HORIZON_SEC:.0f}s)'),
    ]):
        rs, lo_list, hi_list = [], [], []
        for k in keys:
            x = unc_dict[k][mask]
            y = arr[mask]
            r, _, (lo, hi) = pearsonr_ci(x, y, rng=rng)
            rs.append(r)
            lo_list.append(r - lo)
            hi_list.append(hi - r)

        ax.bar(x_pos, rs, color=cols, alpha=0.75, edgecolor='black', lw=0.8)
        ax.errorbar(x_pos, rs, yerr=[lo_list, hi_list],
                    fmt='none', color='black', capsize=4, lw=1.5)
        for xi, (r, lo, hi) in enumerate(zip(rs, lo_list, hi_list)):
            ax.text(xi, r + (hi + 0.005 if r >= 0 else -(lo + 0.005)),
                    f'r={r:.3f}', ha='center',
                    va='bottom' if r >= 0 else 'top',
                    fontsize=8, fontweight='bold')

        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(lbls, fontsize=10)
        ax.set_ylabel('Pearson r', fontsize=10)
        ax.set_title(f'Uncertainty → {title}\n(N={mask.sum():,}, 95% CI)', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'{SUBJECT} — Uncertainty Correlations', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_summary_csv(unc_dict: dict, correct: np.ndarray,
                     future_sw: np.ndarray, actual_action: np.ndarray,
                     out_path: Path):
    keys    = [k for k, *_ in UNCERTAINTY_CFG]
    is_fire = (actual_action == 1) | (actual_action == 4) | (actual_action == 5)
    valid_s = np.isfinite(future_sw)

    rows = []
    for k in keys:
        arr    = unc_dict[k]
        r_acc, p_acc = stats.pearsonr(arr, correct)
        r_sw,  p_sw  = stats.pearsonr(arr[valid_s], future_sw[valid_s])
        rows.append({
            'uncertainty':   k,
            'mean':          float(arr.mean()),
            'std':           float(arr.std()),
            'mean_fire':     float(arr[is_fire].mean()),
            'mean_nofire':   float(arr[~is_fire].mean()),
            'r_accuracy':    float(r_acc),
            'p_accuracy':    float(p_acc),
            'r_switch_rate': float(r_sw),
            'p_switch_rate': float(p_sw),
        })

    fieldnames = list(rows[0].keys())
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (f'{v:.6f}' if isinstance(v, float) else v)
                for k, v in row.items()
            })
    print(f'  Saved -> {out_path}')


# ── Per-run breakdown ─────────────────────────────────────────────────────────

def plot_per_run(records: list[dict], out_path: Path):
    """
    2 rows × n_runs cols.
    Row 0: uncertainty vs action accuracy per run.
    Row 1: uncertainty vs future switch rate per run.
    Each subplot overlays all 3 uncertainty types as separate lines.
    """
    n = len(records)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 8), sharey='row')
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, rec in enumerate(records):
        run_idx  = rec['run_idx']
        unc_dict = {k: rec[k] for k in ('aleatoric', 'epistemic', 'total')}
        valid_s  = np.isfinite(rec['future_sw'])

        # ── row 0: accuracy ───────────────────────────────────────────────────
        ax = axes[0, col]
        for key, color, label in UNCERTAINTY_CFG:
            centers, means, sems, _ = binned_stats(unc_dict[key], rec['correct'], N_BINS)
            r, _ = stats.pearsonr(unc_dict[key], rec['correct'])
            ax.plot(centers, means, 'o-', color=color, lw=1.5, ms=4,
                    label=f'{label[:4]}  r={r:+.2f}')
        ax.axhline(1 / 6, color='black', lw=0.8, ls=':', alpha=0.5, label='chance')
        ax.set_title(
            f'Run {run_idx}\nacc={rec["correct"].mean():.3f}  N={len(rec["correct"]):,}',
            fontsize=8
        )
        ax.set_xlabel('Uncertainty', fontsize=7)
        if col == 0:
            ax.set_ylabel('Action accuracy', fontsize=8)
        ax.legend(fontsize=5.5, loc='upper right')
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

        # ── row 1: switch rate ────────────────────────────────────────────────
        ax = axes[1, col]
        for key, color, label in UNCERTAINTY_CFG:
            x = unc_dict[key][valid_s]
            y = rec['future_sw'][valid_s]
            centers, means, sems, _ = binned_stats(x, y, N_BINS)
            r, _ = stats.pearsonr(x, y)
            ax.plot(centers, means, 'o-', color=color, lw=1.5, ms=4,
                    label=f'{label[:4]}  r={r:+.2f}')
        ax.set_xlabel('Uncertainty', fontsize=7)
        if col == 0:
            ax.set_ylabel(f'Future switch rate\n(next {HORIZON_SEC:.0f}s)', fontsize=8)
        ax.legend(fontsize=5.5, loc='upper right')
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{SUBJECT} — Per-run Uncertainty Analysis', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def save_per_run_csv(records: list[dict], out_path: Path):
    """Per-run stats: accuracy, switch rate, and Pearson r for each uncertainty type."""
    keys = [k for k, *_ in UNCERTAINTY_CFG]
    rows = []
    for rec in records:
        valid_s = np.isfinite(rec['future_sw'])
        row: dict = {
            'run_idx':     rec['run_idx'],
            'n_frames':    len(rec['correct']),
            'accuracy':    float(rec['correct'].mean()),
            'switch_rate': (float(rec['future_sw'][valid_s].mean())
                            if valid_s.any() else float('nan')),
        }
        for k in keys:
            arr = rec[k]
            r_acc, p_acc = stats.pearsonr(arr, rec['correct'])
            r_sw,  p_sw  = stats.pearsonr(arr[valid_s], rec['future_sw'][valid_s])
            row[f'r_{k}_acc'] = float(r_acc)
            row[f'p_{k}_acc'] = float(p_acc)
            row[f'r_{k}_sw']  = float(r_sw)
            row[f'p_{k}_sw']  = float(p_sw)
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (f'{v:.6f}' if isinstance(v, float) else v)
                for k, v in row.items()
            })
    print(f'  Saved -> {out_path}')


# ── State rarity (kNN in DQN 512-dim feature space) ─────────────────────────

def extract_features_512(dqn_model: _DQN,
                         states_uint8: np.ndarray) -> np.ndarray:
    """(N, 4, 84, 84) uint8  →  (N, 512) float64 — post-fc3, pre-fc_out features."""
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states_uint8), BATCH_SIZE):
            x = torch.from_numpy(
                states_uint8[start:start + BATCH_SIZE]
            ).float() / 255.0
            h = F.relu(dqn_model.conv1(x))
            h = F.relu(dqn_model.conv2(h))
            h = F.relu(dqn_model.conv3(h))
            h = h.view(h.size(0), -1)
            h = F.relu(dqn_model.fc3(h))
            chunks.append(h.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


def compute_state_rarity(dqn_model: _DQN,
                         all_npz: list[str],
                         test_run_idx: int,
                         test_states: np.ndarray) -> np.ndarray:
    """
    Mean kNN distance (in normalised 512-dim feature space) from each test frame
    to the reference corpus (all runs except test_run_idx).
    Normalisation uses reference mean / std so test data is never seen during fitting.
    Returns (N_test,) float64.
    """
    from scipy.spatial.distance import cdist

    ref_chunks: list[np.ndarray] = []
    for i, path in enumerate(all_npz):
        if i == test_run_idx:
            continue
        ref_data = np.load(path)
        ref_chunks.append(extract_features_512(dqn_model, ref_data['state']))

    if not ref_chunks:
        return np.full(len(test_states), np.nan)

    ref_feats = np.concatenate(ref_chunks, axis=0)        # (N_ref, 512)
    ref_mean  = ref_feats.mean(axis=0)                    # (512,)
    ref_std   = ref_feats.std(axis=0) + 1e-8              # (512,)
    ref_norm  = (ref_feats - ref_mean) / ref_std          # (N_ref, 512)

    test_feats = extract_features_512(dqn_model, test_states)
    test_norm  = (test_feats - ref_mean) / ref_std        # (N_test, 512)

    CHUNK    = 256
    k_eff    = min(KNN_K, len(ref_norm))
    knn_dist = np.empty(len(test_norm), dtype=np.float64)
    for start in range(0, len(test_norm), CHUNK):
        end        = min(start + CHUNK, len(test_norm))
        D          = cdist(test_norm[start:end], ref_norm)   # (chunk, N_ref)
        knn_dist[start:end] = np.partition(D, k_eff, axis=1)[:, :k_eff].mean(axis=1)

    return knn_dist


def partial_corr(x: np.ndarray, y: np.ndarray,
                 covariate: np.ndarray) -> tuple[float, float, int]:
    """
    Partial Pearson r of x and y after linearly removing covariate from both.
    Returns (r, p, n_valid).
    """
    valid   = np.isfinite(x) & np.isfinite(y) & np.isfinite(covariate)
    x, y, c = x[valid], y[valid], covariate[valid]
    C       = np.column_stack([c, np.ones(len(c))])
    x_res   = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    y_res   = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    r, p    = stats.pearsonr(x_res, y_res)
    return float(r), float(p), int(valid.sum())


def plot_rarity_analysis(unc_dict: dict, correct: np.ndarray,
                         knn_dist: np.ndarray, out_path: Path):
    """
    2 rows × 3 cols.
    Row 0: kNN distance vs each uncertainty type (binned) — rarity↔uncertainty link.
    Row 1: kNN vs accuracy | raw vs partial-r bar | epistemic residual vs accuracy residual.
    """
    keys  = [k for k, *_  in UNCERTAINTY_CFG]
    cols  = [c for _, c, _ in UNCERTAINTY_CFG]
    lbls  = [l for _, _, l in UNCERTAINTY_CFG]
    valid_knn = np.isfinite(knn_dist)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # ── Row 0: kNN vs each uncertainty ───────────────────────────────────────
    for col_i, (key, color, label) in enumerate(UNCERTAINTY_CFG):
        ax    = axes[0, col_i]
        valid = valid_knn & np.isfinite(unc_dict[key])
        x, y  = knn_dist[valid], unc_dict[key][valid]
        ctrs, means, sems, _ = binned_stats(x, y, N_BINS)
        r, p  = stats.pearsonr(x, y)
        ax.fill_between(ctrs, means - sems, means + sems, alpha=0.25, color=color)
        ax.plot(ctrs, means, 'o-', color=color, lw=2, ms=5)
        ax.set_title(f'kNN dist vs {label}\nr={r:.3f}, p={p:.2e}', fontsize=9)
        ax.set_xlabel(f'kNN distance (k={KNN_K})', fontsize=9)
        if col_i == 0:
            ax.set_ylabel('Uncertainty', fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── Row 1, col 0: kNN vs accuracy ────────────────────────────────────────
    ax    = axes[1, 0]
    valid = valid_knn & np.isfinite(correct)
    x, y  = knn_dist[valid], correct[valid]
    ctrs, means, sems, _ = binned_stats(x, y, N_BINS)
    r_knn, p_knn = stats.pearsonr(x, y)
    ax.fill_between(ctrs, means - sems, means + sems, alpha=0.25, color='gray')
    ax.plot(ctrs, means, 'o-', color='gray', lw=2, ms=5)
    ax.axhline(1 / 6, color='black', lw=0.8, ls=':', alpha=0.5, label='chance')
    ax.set_title(f'kNN dist vs Accuracy\nr={r_knn:.3f}, p={p_knn:.2e}', fontsize=9)
    ax.set_xlabel(f'kNN distance (k={KNN_K})', fontsize=9)
    ax.set_ylabel('Action accuracy', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Row 1, col 1: grouped bar — raw r vs partial r ───────────────────────
    ax    = axes[1, 1]
    width = 0.35
    x_pos = np.arange(len(keys))
    raw_rs, par_rs = [], []
    for key in keys:
        r_raw, _           = stats.pearsonr(unc_dict[key], correct)
        r_par, p_par, n_v  = partial_corr(unc_dict[key], correct, knn_dist)
        raw_rs.append(r_raw)
        par_rs.append(r_par)

    for xi, (r_raw, r_par, col) in enumerate(zip(raw_rs, par_rs, cols)):
        ax.bar(xi - width / 2, r_raw, width,
               color=col, alpha=0.85, edgecolor='black', lw=0.8, label='raw r' if xi == 0 else '')
        ax.bar(xi + width / 2, r_par, width,
               color=col, alpha=0.35, edgecolor=col, lw=1.5,
               hatch='//', label='partial r | kNN' if xi == 0 else '')
        ax.text(xi - width / 2, r_raw + (0.005 if r_raw >= 0 else -0.012),
                f'{r_raw:.3f}', ha='center', va='bottom' if r_raw >= 0 else 'top',
                fontsize=7, fontweight='bold')
        ax.text(xi + width / 2, r_par + (0.005 if r_par >= 0 else -0.012),
                f'{r_par:.3f}', ha='center', va='bottom' if r_par >= 0 else 'top',
                fontsize=7, color=col)

    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lbls, fontsize=8)
    ax.set_ylabel('Pearson r (vs Action Accuracy)', fontsize=9)
    ax.set_title('Raw vs Partial r\n(controlling kNN distance)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)

    # ── Row 1, col 2: epistemic residual vs accuracy residual ─────────────────
    ax    = axes[1, 2]
    color = '#d62728'
    valid = np.isfinite(unc_dict['epistemic']) & np.isfinite(correct) & valid_knn
    C     = np.column_stack([knn_dist[valid], np.ones(valid.sum())])
    e_res = (unc_dict['epistemic'][valid]
             - C @ np.linalg.lstsq(C, unc_dict['epistemic'][valid], rcond=None)[0])
    a_res = (correct[valid]
             - C @ np.linalg.lstsq(C, correct[valid], rcond=None)[0])
    ctrs, means, sems, _ = binned_stats(e_res, a_res, N_BINS)
    r_ep, p_ep = stats.pearsonr(e_res, a_res)
    ax.fill_between(ctrs, means - sems, means + sems, alpha=0.25, color=color)
    ax.plot(ctrs, means, 'o-', color=color, lw=2, ms=5)
    ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='black', lw=0.5, ls='--', alpha=0.4)
    ax.set_title(f'Epistemic resid vs Accuracy resid\nr={r_ep:.3f}, p={p_ep:.2e}', fontsize=9)
    ax.set_xlabel('Epistemic uncertainty (kNN removed)', fontsize=9)
    ax.set_ylabel('Accuracy (kNN removed)', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'{SUBJECT} — State Rarity Analysis  (kNN k={KNN_K})',
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def save_rarity_csv(unc_dict: dict, correct: np.ndarray,
                    knn_dist: np.ndarray, out_path: Path):
    valid_knn = np.isfinite(knn_dist)
    rows = []
    for key, _, label in UNCERTAINTY_CFG:
        r_raw, p_raw         = stats.pearsonr(unc_dict[key], correct)
        r_par, p_par, n_v    = partial_corr(unc_dict[key], correct, knn_dist)
        valid_u              = valid_knn & np.isfinite(unc_dict[key])
        r_knn, p_knn         = stats.pearsonr(knn_dist[valid_u], unc_dict[key][valid_u])
        rows.append({
            'uncertainty':   key,
            'r_raw_acc':     float(r_raw),
            'p_raw_acc':     float(p_raw),
            'r_partial_acc': float(r_par),
            'p_partial_acc': float(p_par),
            'r_knn_unc':     float(r_knn),
            'p_knn_unc':     float(p_knn),
            'n_valid':       n_v,
        })
    fieldnames = list(rows[0].keys())
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (f'{v:.6f}' if isinstance(v, float) else v)
                for k, v in row.items()
            })
    print(f'  Saved -> {out_path}')


# ── Run fixed-effect logistic regression ─────────────────────────────────────

def _zscore_run(arr: np.ndarray) -> np.ndarray:
    """Z-score an array, preserving NaN. Returns zeros if std ≈ 0."""
    valid = np.isfinite(arr)
    out   = arr.copy()
    if valid.sum() > 1:
        m = arr[valid].mean()
        s = arr[valid].std()
        out[valid] = (arr[valid] - m) / (s + 1e-12)
    return out


def fit_run_logistic(records: list[dict],
                     out_csv: Path, out_png: Path) -> None:
    """
    Logistic regression of action accuracy with run-wise z-scored predictors
    and run fixed effects.

    Models
    ------
    M1  correct ~ epistemic_z + aleatoric_z + knn_z + C(run)
    M2  correct ~ epistemic_z + aleatoric_z + C(run)   [no kNN, baseline]
    M3  correct ~ total_z     + knn_z       + C(run)   [total instead of decomposed]

    Coefficients are log-odds; plot shows coef ± 1.96*SE (95% CI).
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        print('  statsmodels not found — pip install statsmodels. Skipping regression.')
        return

    # ── Build run-wise z-scored arrays ───────────────────────────────────────
    epi_z_l, alea_z_l, total_z_l, knn_z_l, y_l, run_id_l = [], [], [], [], [], []
    for rec in records:
        n = len(rec['correct'])
        epi_z_l.append(_zscore_run(rec['epistemic']))
        alea_z_l.append(_zscore_run(rec['aleatoric']))
        total_z_l.append(_zscore_run(rec['total']))
        knn_z_l.append(_zscore_run(rec['knn_dist']))
        y_l.append(rec['correct'])
        run_id_l.append(np.full(n, rec['run_idx'], dtype=int))

    epi_z   = np.concatenate(epi_z_l)
    alea_z  = np.concatenate(alea_z_l)
    total_z = np.concatenate(total_z_l)
    knn_z   = np.concatenate(knn_z_l)
    y_full  = np.concatenate(y_l)
    run_ids = np.concatenate(run_id_l)

    # ── Run dummy variables (reference = smallest run_idx) ───────────────────
    sorted_runs  = sorted(set(run_ids.tolist()))
    ref_run      = sorted_runs[0]
    non_ref_runs = sorted_runs[1:]
    run_dummies  = np.column_stack([
        (run_ids == ri).astype(float) for ri in non_ref_runs
    ]) if non_ref_runs else np.empty((len(run_ids), 0), dtype=float)
    dummy_names = [f'run_{ri}_vs_{ref_run}' for ri in non_ref_runs]

    # ── Valid mask (all predictors finite) ───────────────────────────────────
    valid = np.isfinite(epi_z) & np.isfinite(alea_z) & np.isfinite(knn_z)

    def _design(predictor_cols, pred_names):
        cols = [c[valid] for c in predictor_cols]
        if run_dummies.shape[1]:
            cols.append(run_dummies[valid])
        X = sm.add_constant(np.column_stack(cols))
        return X, ['const'] + pred_names + dummy_names

    y = y_full[valid]

    model_specs = [
        ('M1_full',    _design([epi_z, alea_z, knn_z],    ['epistemic_z', 'aleatoric_z', 'knn_z'])),
        ('M2_no_knn',  _design([epi_z, alea_z],           ['epistemic_z', 'aleatoric_z'])),
        ('M3_total',   _design([total_z, knn_z],          ['total_z', 'knn_z'])),
    ]

    # ── Fit & print ───────────────────────────────────────────────────────────
    print(f'\n{"─"*60}')
    print(f'Logistic regression  (N={valid.sum():,}, ref run={ref_run})')
    all_results: list[tuple[str, object, list[str]]] = []
    for model_name, (X, col_names) in model_specs:
        fit = sm.Logit(y, X).fit(disp=False, maxiter=200)
        all_results.append((model_name, fit, col_names))
        pseudo_r2 = 1 - fit.llf / fit.llnull
        print(f'\n  {model_name}  AIC={fit.aic:.1f}  pseudo-R²={pseudo_r2:.4f}')
        for name, coef, se, pval in zip(col_names, fit.params, fit.bse, fit.pvalues):
            if name.startswith('run_') or name == 'const':
                continue
            stars = ('***' if pval < 0.001 else '**' if pval < 0.01
                     else '*' if pval < 0.05 else '')
            print(f'    {name:<20s}: coef={coef:+.4f}  SE={se:.4f}  '
                  f'OR={np.exp(coef):.3f}  p={pval:.3e} {stars}')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows = []
    for model_name, fit, col_names in all_results:
        for name, coef, se, pval in zip(col_names, fit.params, fit.bse, fit.pvalues):
            rows.append({
                'model':   model_name,
                'term':    name,
                'coef':    float(coef),
                'se':      float(se),
                'OR':      float(np.exp(coef)),
                'ci_lo':   float(np.exp(coef - 1.96 * se)),
                'ci_hi':   float(np.exp(coef + 1.96 * se)),
                'p':       float(pval),
                'AIC':     float(fit.aic),
                'pseudoR2': float(1 - fit.llf / fit.llnull),
            })
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v)
                             for k, v in row.items()})
    print(f'  Saved -> {out_csv}')

    # ── Forest plot: key predictors in M1 vs M2, plus knn_z ──────────────────
    focus_terms  = ['epistemic_z', 'aleatoric_z', 'knn_z']
    focus_labels = ['Epistemic', 'Aleatoric', 'kNN dist']
    colors_m     = {'M1_full': '#d62728', 'M2_no_knn': '#1f77b4'}
    model_labels = {'M1_full': 'M1 (epi+alea+kNN)', 'M2_no_knn': 'M2 (epi+alea)'}

    fig, ax = plt.subplots(figsize=(7, 5))
    n_terms    = len(focus_terms)
    y_base     = np.arange(n_terms, dtype=float)
    offsets    = {'M1_full': 0.15, 'M2_no_knn': -0.15}
    drawn_labels: set[str] = set()

    for model_name, fit, col_names in all_results:
        if model_name not in colors_m:
            continue
        col_map = dict(zip(col_names, zip(fit.params, fit.bse, fit.pvalues)))
        for i, term in enumerate(focus_terms):
            if term not in col_map:
                continue
            coef, se, pval = col_map[term]
            ci_lo, ci_hi   = coef - 1.96 * se, coef + 1.96 * se
            ypos           = y_base[i] + offsets[model_name]
            lbl            = model_labels[model_name] if model_name not in drawn_labels else ''
            ax.errorbar(coef, ypos, xerr=[[coef - ci_lo], [ci_hi - coef]],
                        fmt='o', color=colors_m[model_name], ms=7, lw=2,
                        capsize=4, label=lbl)
            drawn_labels.add(model_name)
            stars = ('***' if pval < 0.001 else '**' if pval < 0.01
                     else '*' if pval < 0.05 else 'ns')
            ax.text(ci_hi + 0.005, ypos, stars, va='center', fontsize=8,
                    color=colors_m[model_name])

    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_yticks(y_base)
    ax.set_yticklabels(focus_labels, fontsize=11)
    ax.set_xlabel('Log-odds coefficient (95% CI)', fontsize=10)
    ax.set_title(
        f'{SUBJECT} — Fixed-effects Logistic Regression\n'
        f'(run-wise z-scored, ref run={ref_run}, N={valid.sum():,})',
        fontsize=10
    )
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_png}')


# ── NOOP event analysis (one data point per NOOP run) ────────────────────────

def _iter_runs(actual_action: np.ndarray):
    """Yield (start, end_excl, action) for each consecutive same-action run."""
    i = 0
    while i < len(actual_action):
        j = i + 1
        while j < len(actual_action) and actual_action[j] == actual_action[i]:
            j += 1
        yield i, j, int(actual_action[i])
        i = j


def compute_noop_events(actual_action: np.ndarray,
                        unc_dict: dict) -> tuple[list[dict], dict]:
    """
    One data point per consecutive NOOP (action==0) run.

    Returns
    -------
    noop_events : list of dicts
        length, onset_{key}, mean_{key}, pre_{key}, post_{key} per uncertainty key.
    onset_unc : dict  {key: {'noop': array, 'non_noop': array}}
        Onset uncertainty for every action run, split by NOOP / non-NOOP (Q.A).
    """
    n = len(actual_action)
    noop_events   = []
    noop_onset    = {k: [] for k in unc_dict}
    non_noop_onset = {k: [] for k in unc_dict}

    for run_start, run_end, action in _iter_runs(actual_action):
        for k, arr in unc_dict.items():
            onset_val = float(arr[run_start])
            (noop_onset if action == 0 else non_noop_onset)[k].append(onset_val)

        if action != 0:
            continue

        event = {'length': run_end - run_start, 'start': run_start, 'end': run_end}
        for k, arr in unc_dict.items():
            event[f'onset_{k}'] = float(arr[run_start])
            event[f'mean_{k}']  = float(arr[run_start:run_end].mean())
            pre_s = max(0, run_start - PRE_POST_FRAMES)
            event[f'pre_{k}']   = (float(arr[pre_s:run_start].mean())
                                   if run_start > 0 else np.nan)
            post_e = min(n, run_end + PRE_POST_FRAMES)
            event[f'post_{k}']  = (float(arr[run_end:post_e].mean())
                                   if run_end < n else np.nan)
        noop_events.append(event)

    onset_unc = {
        k: {
            'noop':     np.array(noop_onset[k]),
            'non_noop': np.array(non_noop_onset[k]),
        }
        for k in unc_dict
    }
    return noop_events, onset_unc


def plot_noop_event_analysis(noop_events: list[dict], onset_unc: dict,
                              out_path: Path):
    """
    3 rows × 3 cols (one col per uncertainty type).

    Row 0  Q.A  Does high uncertainty trigger NOOP onset?
               Violin: onset uncertainty at NOOP vs non-NOOP transitions.
    Row 1  Q.B  Does high uncertainty predict longer NOOP?
               Binned line: NOOP run length (x) vs onset uncertainty (y).
    Row 2  Q.C  Does uncertainty change around NOOP?
               Bar: mean pre / during / post NOOP for each uncertainty type.
    """
    keys  = [k for k, *_ in UNCERTAINTY_CFG]
    n_ev  = len(noop_events)

    # Pre-compute per-event arrays
    ev_len    = np.array([e['length'] for e in noop_events], dtype=float)
    ev_vals   = {k: {
        'onset': np.array([e[f'onset_{k}'] for e in noop_events]),
        'mean':  np.array([e[f'mean_{k}']  for e in noop_events]),
        'pre':   np.array([e[f'pre_{k}']   for e in noop_events]),
        'post':  np.array([e[f'post_{k}']  for e in noop_events]),
    } for k in keys}

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for col, (key, color, label) in enumerate(UNCERTAINTY_CFG):
        # ── Q.A: NOOP onset vs non-NOOP onset violin ──────────────────────────
        ax = axes[0, col]
        noop_on     = onset_unc[key]['noop']
        non_noop_on = onset_unc[key]['non_noop']
        vp = ax.violinplot([non_noop_on, noop_on], positions=[0, 1],
                           showmedians=True, showextrema=False)
        for patch, c in zip(vp['bodies'], ['#aec7e8', color]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        vp['cmedians'].set_color('black')
        _, p_mw = stats.mannwhitneyu(noop_on, non_noop_on,
                                     alternative='two-sided')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['non-NOOP\nonset', 'NOOP\nonset'], fontsize=8)
        ax.set_title(
            f'{label}\nMW p={p_mw:.2e}  '
            f'Δmedian={np.median(noop_on)-np.median(non_noop_on):+.4f}',
            fontsize=8
        )
        ax.set_ylabel('Onset uncertainty', fontsize=8) if col == 0 else None
        ax.grid(True, axis='y', alpha=0.3)

        # ── Q.B: run length vs onset uncertainty ──────────────────────────────
        ax = axes[1, col]
        x = ev_len
        y = ev_vals[key]['onset']
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() > 1:
            centers, means, sems, _ = binned_stats(x[valid], y[valid], N_BINS)
            r, p = stats.pearsonr(x[valid], y[valid])
            ax.fill_between(centers, means - sems, means + sems,
                            alpha=0.25, color=color)
            ax.plot(centers, means, 'o-', color=color, lw=2, ms=5)
            ax.set_title(f'{label}\nr={r:.3f}, p={p:.2e}', fontsize=8)
        ax.set_xlabel('NOOP run length (frames)', fontsize=8)
        ax.set_ylabel('Onset uncertainty', fontsize=8) if col == 0 else None
        ax.grid(True, alpha=0.3)

        # ── Q.C: pre / during / post ──────────────────────────────────────────
        ax = axes[2, col]
        conditions = [
            ('pre',   f'Pre\n({PRE_POST_FRAMES}f)', '#aec7e8'),
            ('mean',  'During',                      color),
            ('post',  f'Post\n({PRE_POST_FRAMES}f)', '#ffbb78'),
        ]
        x_pos  = np.arange(len(conditions))
        vals   = []
        errs   = []
        for cond_key, _, _ in conditions:
            v = ev_vals[key][cond_key]
            v = v[np.isfinite(v)]
            vals.append(v.mean() if len(v) else np.nan)
            errs.append(stats.sem(v) if len(v) > 1 else 0.0)

        bar_colors = [c for _, _, c in conditions]
        ax.bar(x_pos, vals, yerr=errs, color=bar_colors, alpha=0.8,
               edgecolor='black', lw=0.8, capsize=4)
        # paired t-test pre vs during, during vs post
        for (ia, ib), yoff in [((0, 1), max(vals) * 1.05), ((1, 2), max(vals) * 1.10)]:
            va = ev_vals[key][conditions[ia][0]]
            vb = ev_vals[key][conditions[ib][0]]
            both = np.isfinite(va) & np.isfinite(vb)
            if both.sum() > 1:
                _, pt = stats.ttest_rel(va[both], vb[both])
                ax.annotate('', xy=(ib, yoff), xytext=(ia, yoff),
                            arrowprops=dict(arrowstyle='-', color='black', lw=0.8))
                ax.text((ia + ib) / 2, yoff * 1.01, f'p={pt:.2e}',
                        ha='center', va='bottom', fontsize=6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([lbl for _, lbl, _ in conditions], fontsize=8)
        ax.set_title(f'{label}\n(N_events={n_ev:,})', fontsize=8)
        ax.set_ylabel('Uncertainty', fontsize=8) if col == 0 else None
        ax.grid(True, axis='y', alpha=0.3)

    row_labels = [
        'Q.A  NOOP onset vs non-NOOP onset',
        'Q.B  NOOP run length vs onset uncertainty',
        'Q.C  Pre / During / Post NOOP',
    ]
    for row, lbl in enumerate(row_labels):
        axes[row, 0].annotate(lbl, xy=(0, 0.5), xytext=(-0.35, 0.5),
                               xycoords='axes fraction', textcoords='axes fraction',
                               fontsize=9, fontweight='bold', rotation=90,
                               ha='center', va='center')

    fig.suptitle(
        f'{SUBJECT} — NOOP Event Analysis  '
        f'(N_events={n_ev:,}, pre/post={PRE_POST_FRAMES} frames)',
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def save_noop_event_csv(noop_events: list[dict], out_path: Path):
    """Per-NOOP-event CSV + summary stats for Q.A, Q.B, Q.C."""
    keys = [k for k, *_ in UNCERTAINTY_CFG]
    # Per-event rows
    fieldnames = ['event_idx', 'start', 'end', 'length']
    for k in keys:
        fieldnames += [f'onset_{k}', f'mean_{k}', f'pre_{k}', f'post_{k}']

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, ev in enumerate(noop_events):
            row = {'event_idx': idx, 'start': ev['start'],
                   'end': ev['end'], 'length': ev['length']}
            for k in keys:
                for pfx in ('onset', 'mean', 'pre', 'post'):
                    v = ev.get(f'{pfx}_{k}', np.nan)
                    row[f'{pfx}_{k}'] = f'{v:.6f}' if np.isfinite(v) else ''
            writer.writerow(row)
    print(f'  Saved -> {out_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--calibrate', action=argparse.BooleanOptionalAction,
                   default=False,
                   help='Calibrate softmax temperature T per model using all other '
                        'runs in RUNS as calibration data. '
                        '--no-calibrate (default) uses T=1.0.')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_npz = sorted(glob.glob(str(DATA_DIR / '*.npz')))
    if not all_npz:
        raise FileNotFoundError(f'No npz files in {DATA_DIR}')

    print(f'Subject   : {SUBJECT}')
    print(f'Runs      : {[ri for ri, _ in RUNS]}')
    print(f'AGG       : {AGG}')
    print(f'Calibrate : {args.calibrate}')

    dqn_model = load_dqn(DQN_PATH)

    # ── Process each run ──────────────────────────────────────────────────────
    records: list[dict] = []
    for run_idx, model_fname in RUNS:
        if run_idx >= len(all_npz):
            print(f'  WARNING: run_idx={run_idx} out of range '
                  f'({len(all_npz)} files) — skip')
            continue

        model_path = MODEL_DIR / model_fname
        if not model_path.exists():
            print(f'  WARNING: model not found: {model_path} — skip')
            continue

        npz_path = Path(all_npz[run_idx])
        print(f'\n[run {run_idx}]  {npz_path.name}  |  model: {model_fname}')

        heads = load_ensemble_heads(model_path, dqn_model)

        # Phase 1: calibrate T on all other runs in RUNS
        if args.calibrate:
            other_npz = [all_npz[ri] for ri, _ in RUNS
                         if ri != run_idx and ri < len(all_npz)]
            if other_npz:
                T_cal, nll_t1, nll_opt = calibrate_temperature(
                    heads, dqn_model, other_npz, agg=AGG)
                print(f'  T_cal={T_cal:.4f}  '
                      f'(NLL: T=1→{nll_t1:.4f}, T*→{nll_opt:.4f})')
            else:
                T_cal = 1.0
                print('  T_cal=1.0 (no other runs available for calibration)')
        else:
            T_cal = 1.0

        # Phase 2: inference
        data          = np.load(npz_path)
        time_         = data['time'].astype(np.float64)
        states        = data['state']
        acts          = data['action']
        actual_action = acts.argmax(axis=1) if acts.ndim == 2 else acts.astype(int)

        print(f'  Frames: {len(time_):,}  |  Computing Q-values...')
        q_all = extract_q_all(heads, dqn_model, states)

        aleatoric, epistemic, total = compute_uncertainty(q_all, temp=T_cal, agg=AGG)

        print(f'  Computing state rarity (kNN k={KNN_K})...')
        knn_dist = compute_state_rarity(dqn_model, all_npz, run_idx, states)

        # Predicted action (argmax of aggregate Q, invariant to mean-centering)
        q_agg       = q_all.mean(axis=1) if AGG == 'mean' else q_all.min(axis=1)
        pred_action = q_agg.argmax(axis=1).astype(int)
        correct     = (pred_action == actual_action).astype(np.float64)

        switch    = compute_switch_indicator(actual_action)
        future_sw = future_window_mean(time_, switch, HORIZON_SEC)

        valid_s = np.isfinite(future_sw)
        r_acc, _ = stats.pearsonr(aleatoric, correct)
        r_sw,  _ = stats.pearsonr(aleatoric[valid_s], future_sw[valid_s])
        print(f'  action accuracy : {correct.mean():.4f}')
        print(f'  switch rate     : {switch.mean():.4f}')
        print(f'  aleatoric       : mean={aleatoric.mean():.4f}  '
              f'r(acc)={r_acc:+.4f}  r(sw)={r_sw:+.4f}')

        records.append({
            'run_idx':       run_idx,
            'aleatoric':     aleatoric,
            'epistemic':     epistemic,
            'total':         total,
            'actual_action': actual_action,
            'correct':       correct,
            'future_sw':     future_sw,
            'knn_dist':      knn_dist,
        })

    if not records:
        print('ERROR: no data collected.')
        return

    # ── Pool all records ──────────────────────────────────────────────────────
    unc_dict = {
        'aleatoric': np.concatenate([r['aleatoric'] for r in records]),
        'epistemic': np.concatenate([r['epistemic'] for r in records]),
        'total':     np.concatenate([r['total']     for r in records]),
    }
    correct       = np.concatenate([r['correct']       for r in records])
    future_sw     = np.concatenate([r['future_sw']     for r in records])
    actual_action = np.concatenate([r['actual_action'] for r in records])
    knn_dist      = np.concatenate([r['knn_dist']      for r in records])

    valid_s = np.isfinite(future_sw)
    print(f'\n{"="*60}')
    print(f'Runs collected  : {[r["run_idx"] for r in records]}')
    print(f'Total frames    : {len(correct):,}')
    print(f'Action accuracy : {correct.mean():.4f}')
    for key, _, label in UNCERTAINTY_CFG:
        x = unc_dict[key]
        r_acc,  p_acc  = stats.pearsonr(x, correct)
        r_sw,   p_sw   = stats.pearsonr(x[valid_s], future_sw[valid_s])
        r_par,  p_par, _ = partial_corr(x, correct, knn_dist)
        print(f'  {label:<18s}: mean={x.mean():.4f}  '
              f'r(acc)={r_acc:+.4f} p={p_acc:.2e}  '
              f'r(acc|kNN)={r_par:+.4f} p={p_par:.2e}  '
              f'r(sw)={r_sw:+.4f} p={p_sw:.2e}')

    noop_events, onset_unc = compute_noop_events(actual_action, unc_dict)
    print(f'NOOP events     : {len(noop_events):,}  '
          f'(mean length={np.mean([e["length"] for e in noop_events]):.1f} frames)')
    for key, _, label in UNCERTAINTY_CFG:
        lengths    = np.array([e['length']        for e in noop_events], float)
        onsets     = np.array([e[f'onset_{key}']  for e in noop_events], float)
        r_b, p_b   = stats.pearsonr(lengths, onsets)
        noop_on    = onset_unc[key]['noop']
        non_on     = onset_unc[key]['non_noop']
        _, p_mw    = stats.mannwhitneyu(noop_on, non_on, alternative='two-sided')
        print(f'  {label:<18s}: '
              f'Q.A MW p={p_mw:.2e}  '
              f'Q.B r(len,onset)={r_b:+.3f} p={p_b:.2e}')

    # ── Generate plots ────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    plot_uncertainty_dist(
        unc_dict, actual_action,
        OUT_DIR / 'uncertainty_dist.png')
    plot_action_accuracy_binned(
        unc_dict, correct,
        OUT_DIR / 'action_accuracy_binned.png')
    plot_action_switch_binned(
        unc_dict, future_sw,
        OUT_DIR / 'action_switch_binned.png')
    plot_scatter_alea_epi(
        unc_dict, future_sw,
        OUT_DIR / 'scatter_alea_epi.png')
    plot_corr_bar(
        unc_dict, correct, future_sw,
        OUT_DIR / 'corr_bar.png')
    save_summary_csv(
        unc_dict, correct, future_sw, actual_action,
        OUT_DIR / 'summary_stats.csv')
    plot_per_run(
        records,
        OUT_DIR / 'per_run.png')
    save_per_run_csv(
        records,
        OUT_DIR / 'per_run_stats.csv')
    plot_noop_event_analysis(
        noop_events, onset_unc,
        OUT_DIR / 'noop_event_analysis.png')
    save_noop_event_csv(
        noop_events,
        OUT_DIR / 'noop_events.csv')
    plot_rarity_analysis(
        unc_dict, correct, knn_dist,
        OUT_DIR / 'rarity_analysis.png')
    save_rarity_csv(
        unc_dict, correct, knn_dist,
        OUT_DIR / 'rarity_analysis.csv')
    fit_run_logistic(
        records,
        OUT_DIR / 'logistic_regression.csv',
        OUT_DIR / 'logistic_regression.png')

    print(f'\nDone.  All outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
