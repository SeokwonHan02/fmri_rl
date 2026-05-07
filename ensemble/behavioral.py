"""
behavioral.py

Ensemble DQN uncertainty analysis for sub_2 run 10.

Loads the ensemble model trained on sub_2 (all runs except run 10)
and applies it to the held-out run 10 data.  Per-frame quantities:
  - Aleatoric uncertainty  (mean within-head entropy)
  - Epistemic uncertainty  (cross-head disagreement)
  - Total predictive entropy

Analyses & plots (saved to ensemble/results/):
  uncertainty_timeseries.png   — 4-panel time series
  future_reward_binned.png     — 3-panel binned mean ± SEM (unc vs future reward)
  future_reward_lag.png        — lag curve (Pearson r, lags -6…+6 s)
  action_switch_binned.png     — 3-panel binned future switch rate
  action_switch_scatter.png    — aleatoric vs epistemic scatter colored by switch rate
  uncertainty_dist.png         — distribution split by FIRE / non-FIRE
  correlation_bar.png          — Pearson r bar with 95 % CI bootstrap
  summary_stats.csv
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import softmax as _softmax

_HERE = Path(__file__).resolve().parent   # ensemble/
_ROOT = _HERE.parent                       # fMRI_RL/

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = _ROOT / 'processed_data_frameskip_4' / 'sub_6'
MODEL_PATH  = _HERE / 'trained_ensemble_model' / 'sub_6_run_10.pth'
DQN_PATH    = _ROOT / 'pretrained' / 'dqn_cnn.pt'
OUT_DIR     = _HERE / 'results'
SUBJECT     = 'sub_6'
RUN_IDX     = 10           # 0-based index of the held-out run
HORIZON_SEC = 5.0          # future window for reward / switch analyses
N_BINS      = 10
EPS         = 1e-12
BATCH_SIZE  = 512

# Direction group: NOOP/FIRE→0, RIGHT/RIGHT+FIRE→1, LEFT/LEFT+FIRE→2
_ACTION_TO_DIR = np.array([0, 0, 1, 2, 1, 2], dtype=np.int32)

# ── DQN model definition ──────────────────────────────────────────────────────
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

def _robust_load(path: str, map_location='cpu'):
    """Load a checkpoint, falling back to weights_only=False on older PyTorch."""
    p = str(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(p, map_location=map_location)


def load_dqn(path: Path) -> _DQN:
    ckpt = _robust_load(path)
    sd   = ckpt.get('policy_net', ckpt)   # checkpoint may wrap weights under 'policy_net'
    dqn  = _DQN()
    dqn.load_state_dict(sd)
    dqn.eval()
    print(f'  DQN loaded: {path}')
    return dqn


def load_ensemble_heads(model_path: Path, dqn_model: _DQN) -> nn.ModuleList:
    """
    Load ensemble head weights.  File contains the state_dict of an
    nn.ModuleList whose keys follow '{head_idx}.{layer_idx}.weight/bias'.
    """
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
    print(f'  Ensemble heads: {num_heads}')

    heads = nn.ModuleList([
        nn.Sequential(nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, 6))
        for _ in range(num_heads)
    ])
    heads.load_state_dict(heads_sd)
    heads.eval()
    print(f'  Ensemble model loaded: {model_path}')
    return heads


# ── Inference ─────────────────────────────────────────────────────────────────

def extract_q_all(heads: nn.ModuleList,
                  dqn_model: _DQN,
                  states_uint8: np.ndarray) -> np.ndarray:
    """
    Forward pass through frozen CNN + each head.
    states_uint8 : (N, 4, 84, 84) uint8
    returns       : (N, H, 6) float64
    """
    N = len(states_uint8)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, N, BATCH_SIZE):
            x = torch.from_numpy(
                states_uint8[start:start + BATCH_SIZE]
            ).float() / 255.0
            h = F.relu(dqn_model.conv1(x))
            h = F.relu(dqn_model.conv2(h))
            h = F.relu(dqn_model.conv3(h))
            feat  = h.view(h.size(0), -1)                         # (B, 3136)
            q_all = torch.stack([hd(feat) for hd in heads], dim=1)  # (B, H, 6)
            chunks.append(q_all.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)      # (N, H, 6)


# ── Uncertainty decomposition ─────────────────────────────────────────────────

def compute_uncertainty(q_all: np.ndarray):
    """
    PEN decomposition.
    q_all : (N, H, 6)
    returns: aleatoric (N,), epistemic (N,), total (N,)  all float64
    """
    probs_all  = _softmax(q_all, axis=2)                                       # (N, H, 6)
    probs_mean = probs_all.mean(axis=1)                                        # (N, 6)
    total      = -(probs_mean * np.log(probs_mean + EPS)).sum(axis=1)         # (N,)
    aleatoric  = -(probs_all  * np.log(probs_all  + EPS)).sum(axis=2).mean(axis=1)  # (N,)
    epistemic  = total - aleatoric
    return aleatoric.astype(np.float64), epistemic.astype(np.float64), total.astype(np.float64)


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
    N      = len(time)
    cumval = np.concatenate([[0.0], np.cumsum(values)])
    j_end  = np.searchsorted(time, time + horizon, side='right')
    j_start = np.arange(N) + 1
    counts  = (j_end - j_start).astype(np.float64)
    total   = cumval[j_end] - cumval[j_start]
    return np.where(counts > 0, total / counts, np.nan).astype(np.float64)


def future_reward(time: np.ndarray, reward: np.ndarray,
                  horizon: float) -> np.ndarray:
    """Sum of rewards in (time[i], time[i]+horizon]."""
    N      = len(time)
    cumrew = np.concatenate([[0.0], np.cumsum(reward)])
    j_hi   = np.searchsorted(time, time + horizon, side='right')
    j_lo   = np.arange(N) + 1
    return (cumrew[j_hi] - cumrew[j_lo]).astype(np.float64)


def windowed_reward_at_lag(time: np.ndarray, reward: np.ndarray,
                           lag: float, window: float) -> np.ndarray:
    """Sum of rewards in (time[i]+lag, time[i]+lag+window]."""
    N      = len(time)
    cumrew = np.concatenate([[0.0], np.cumsum(reward)])
    t_hi   = time + lag + window
    t_lo   = time + lag
    j_hi   = np.searchsorted(time, t_hi, side='right')
    j_lo   = np.searchsorted(time, t_lo, side='right')
    return (cumrew[j_hi] - cumrew[j_lo]).astype(np.float64)


# ── Statistics helpers ────────────────────────────────────────────────────────

def binned_stats(x: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Percentile-binned mean ± SEM."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    edges      = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    edges[-1] += 1e-9
    centers, means, sems, ns = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m    = (x >= lo) & (x < hi)
        vals = y[m]
        centers.append(np.median(x[m]) if m.any() else (lo + hi) / 2)
        means.append(vals.mean()           if len(vals)     else np.nan)
        sems.append(stats.sem(vals)        if len(vals) > 1 else np.nan)
        ns.append(m.sum())
    return (np.array(centers), np.array(means),
            np.array(sems),    np.array(ns, dtype=int))


def pearsonr_ci(x: np.ndarray, y: np.ndarray,
                n_boot: int = 1000, ci: float = 0.95,
                rng: np.random.Generator | None = None):
    """Pearson r with bootstrap CI.  Returns r, p, (lo, hi)."""
    r, p = stats.pearsonr(x, y)
    if rng is None:
        rng = np.random.default_rng(42)
    n    = len(x)
    rs   = np.array([
        stats.pearsonr(x[idx := rng.integers(0, n, n)], y[idx])[0]
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.nanpercentile(rs, [100 * alpha, 100 * (1 - alpha)])
    return r, p, (lo, hi)


# ── Plotting ──────────────────────────────────────────────────────────────────

UNCERTAINTY_CFG = [
    ('aleatoric', '#1f77b4', 'Aleatoric (within-head entropy)'),
    ('epistemic', '#d62728', 'Epistemic (cross-head disagreement)'),
    ('total',     '#ff7f0e', 'Total predictive entropy'),
]


def plot_uncertainty_timeseries(time, aleatoric, epistemic, total, reward,
                                 out_path: Path):
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    for ax, arr, label, color in [
        (axes[0], aleatoric, 'Aleatoric', '#1f77b4'),
        (axes[1], epistemic, 'Epistemic', '#d62728'),
        (axes[2], total,     'Total entropy', '#ff7f0e'),
    ]:
        ax.fill_between(time, 0, arr, alpha=0.3, color=color, linewidth=0)
        ax.plot(time, arr, lw=0.6, color=color)
        ax.axhline(np.log(6), color='gray', lw=0.7, ls='--', alpha=0.5,
                   label=f'max=ln(6)={np.log(6):.3f}')
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)

    # reward panel: event markers
    rpos = reward > 0
    axes[3].vlines(time[rpos], 0, reward[rpos], color='#2ca02c', lw=0.8, alpha=0.7)
    axes[3].axhline(0, color='black', lw=0.5)
    axes[3].set_ylabel('Reward', fontsize=9)
    axes[3].set_xlabel('Game time (s)', fontsize=9)
    axes[3].grid(True, alpha=0.2)

    fig.suptitle(f'{SUBJECT} run {RUN_IDX} — Uncertainty Time Series', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def plot_future_reward_binned(unc_dict: dict, fut_rew: np.ndarray,
                               horizon: float, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    valid = np.isfinite(fut_rew)

    for ax, (key, color, label) in zip(axes, UNCERTAINTY_CFG):
        x = unc_dict[key][valid]
        y = fut_rew[valid]
        centers, means, sems, ns = binned_stats(x, y, N_BINS)
        r, p = stats.pearsonr(x, y)

        ax.fill_between(centers, means - sems, means + sems,
                        alpha=0.25, color=color)
        ax.plot(centers, means, 'o-', color=color, lw=2, ms=5)
        ax.axvline(np.log(6), color='gray', lw=0.8, ls='--', alpha=0.5,
                   label=f'max=ln(6)')
        ax.set_title(f'{label}\nr={r:.3f}, p={p:.2e}', fontsize=9)
        ax.set_xlabel('Uncertainty', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f'Future reward (next {horizon:.0f}s)', fontsize=9)
    fig.suptitle(
        f'{SUBJECT} run {RUN_IDX} — Uncertainty vs Future Reward  (N={valid.sum():,})',
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def plot_future_reward_lag(unc_dict: dict, time: np.ndarray, reward: np.ndarray,
                            lags: list, window: float, out_path: Path):
    lag_arr  = np.array(lags)
    rs_dict  = {k: [] for k, *_ in UNCERTAINTY_CFG}

    for lag in lags:
        wr   = windowed_reward_at_lag(time, reward, lag, window)
        mask = np.isfinite(wr)
        for k, *_ in UNCERTAINTY_CFG:
            x = unc_dict[k][mask]
            y = wr[mask]
            r, _ = stats.pearsonr(x, y) if len(x) > 5 else (np.nan, np.nan)
            rs_dict[k].append(r)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, color, label in UNCERTAINTY_CFG:
        ax.plot(lag_arr, rs_dict[key], 'o-', color=color, lw=2, ms=5, label=label)

    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axvline(0, color='#cc4444', lw=1, ls=':', alpha=0.7, label='lag=0')
    ax.set_xlabel('Lag (s)  — positive = future reward window', fontsize=10)
    ax.set_ylabel('Pearson r', fontsize=10)
    ax.set_title(
        f'{SUBJECT} run {RUN_IDX} — Uncertainty–Reward Lag Curve\n'
        f'window={window:.1f}s',
        fontsize=11
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def plot_action_switch_binned(unc_dict: dict, future_sw: np.ndarray,
                               horizon: float, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    valid     = np.isfinite(future_sw)

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

    axes[0].set_ylabel(f'Future action switch rate\n(next {horizon:.0f}s)', fontsize=9)
    fig.suptitle(
        f'{SUBJECT} run {RUN_IDX} — Uncertainty vs Future Action Switch Rate  '
        f'(N={valid.sum():,})',
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def plot_action_switch_scatter(unc_dict: dict, future_sw: np.ndarray,
                                horizon: float, out_path: Path):
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
    plt.colorbar(sc, ax=ax, label=f'Future switch rate (next {horizon:.0f}s)')
    ax.set_xlabel('Aleatoric uncertainty', fontsize=10)
    ax.set_ylabel('Epistemic uncertainty', fontsize=10)
    ax.set_title(
        f'{SUBJECT} run {RUN_IDX} — Aleatoric vs Epistemic\n'
        f'(color = switch rate, shown {len(alea):,} pts)',
        fontsize=10
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def plot_uncertainty_dist(unc_dict: dict, actual_action: np.ndarray,
                           out_path: Path):
    """Aleatoric / epistemic / total distributions split by FIRE vs non-FIRE."""
    is_fire  = (actual_action == 1) | (actual_action == 4) | (actual_action == 5)
    is_nofire = ~is_fire

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, color, label) in zip(axes, UNCERTAINTY_CFG):
        arr = unc_dict[key]
        ax.hist(arr[is_fire],  bins=60, alpha=0.55, color='#d62728',
                density=True, label=f'FIRE (n={is_fire.sum():,})', linewidth=0)
        ax.hist(arr[is_nofire], bins=60, alpha=0.45, color='#1f77b4',
                density=True, label=f'non-FIRE (n={is_nofire.sum():,})', linewidth=0)
        ax.axvline(np.log(6), color='black', lw=1, ls='--',
                   label=f'max=ln(6)={np.log(6):.3f}')

        # KS test
        ks, ks_p = stats.ks_2samp(arr[is_fire], arr[is_nofire])
        ax.set_title(f'{label}\nKS={ks:.3f}, p={ks_p:.2e}', fontsize=9)
        ax.set_xlabel('Uncertainty', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'{SUBJECT} run {RUN_IDX} — Uncertainty Distribution: FIRE vs non-FIRE',
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


def plot_correlation_bar(unc_dict: dict, fut_rew: np.ndarray,
                          future_sw: np.ndarray, out_path: Path):
    """
    2-panel bar: Pearson r of each uncertainty vs (future reward, future switch rate).
    Error bars = 95 % bootstrap CI.
    """
    rng   = np.random.default_rng(42)
    keys  = [k for k, *_ in UNCERTAINTY_CFG]
    cols  = [c for _, c, _ in UNCERTAINTY_CFG]
    lbls  = [l.split(' ')[0] for _, _, l in UNCERTAINTY_CFG]
    x_pos = np.arange(len(keys))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (outcome_arr, title) in zip(axes, [
        (fut_rew,   f'Future Reward (next {HORIZON_SEC:.0f}s)'),
        (future_sw, f'Future Switch Rate (next {HORIZON_SEC:.0f}s)'),
    ]):
        valid = np.isfinite(outcome_arr)
        rs, lo_list, hi_list = [], [], []
        for k in keys:
            x = unc_dict[k][valid]
            y = outcome_arr[valid]
            r, _, (lo, hi) = pearsonr_ci(x, y, n_boot=1000, rng=rng)
            rs.append(r)
            lo_list.append(r - lo)
            hi_list.append(hi - r)

        ax.bar(x_pos, rs, color=cols, alpha=0.75, edgecolor='black', lw=0.8)
        ax.errorbar(x_pos, rs,
                    yerr=[lo_list, hi_list],
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
        ax.set_title(f'Uncertainty → {title}\n(N={valid.sum():,}, 95% CI)', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'{SUBJECT} run {RUN_IDX} — Uncertainty Correlations', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_summary_csv(unc_dict: dict, fut_rew: np.ndarray,
                      future_sw: np.ndarray, actual_action: np.ndarray,
                      out_path: Path):
    import csv

    keys    = [k for k, *_ in UNCERTAINTY_CFG]
    is_fire = (actual_action == 1) | (actual_action == 4) | (actual_action == 5)
    valid_r = np.isfinite(fut_rew)
    valid_s = np.isfinite(future_sw)

    rows = []
    for k in keys:
        arr = unc_dict[k]
        r_r, p_r = stats.pearsonr(arr[valid_r], fut_rew[valid_r])
        r_s, p_s = stats.pearsonr(arr[valid_s], future_sw[valid_s])
        rows.append({
            'uncertainty': k,
            'mean':        float(arr.mean()),
            'std':         float(arr.std()),
            'min':         float(arr.min()),
            'max':         float(arr.max()),
            'mean_fire':   float(arr[is_fire].mean()),
            'mean_nofire': float(arr[~is_fire].mean()),
            'r_future_reward': float(r_r),
            'p_future_reward': float(p_r),
            'r_future_switch': float(r_s),
            'p_future_switch': float(p_s),
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Locate run 10 data file ────────────────────────────────────────────
    import glob
    npz_files = sorted(glob.glob(str(DATA_DIR / '*.npz')))
    if not npz_files:
        raise FileNotFoundError(f'No npz files in {DATA_DIR}')
    if RUN_IDX >= len(npz_files):
        raise ValueError(f'RUN_IDX={RUN_IDX} but only {len(npz_files)} files found')

    npz_path = Path(npz_files[RUN_IDX])
    print(f'Subject  : {SUBJECT}')
    print(f'Run file : {npz_path.name}  (index {RUN_IDX})')
    print(f'Model    : {MODEL_PATH}')
    print(f'Out dir  : {OUT_DIR}')

    data = np.load(npz_path)
    time_   = data['time'].astype(np.float64)          # (N,)
    states  = data['state']                             # (N, 4, 84, 84) uint8
    acts    = data['action']                            # (N, 6) one-hot
    reward  = data['reward'].astype(np.float64)        # (N,)
    N       = len(time_)

    actual_action = (acts.argmax(axis=1) if acts.ndim == 2
                     else acts.astype(int))             # (N,)

    print(f'Frames   : {N:,}  |  time: {time_[0]:.2f}s – {time_[-1]:.2f}s')
    print(f'Actions  : {np.unique(actual_action, return_counts=True)}')

    # ── 2. Load models ────────────────────────────────────────────────────────
    print('\nLoading DQN backbone...')
    dqn_model = load_dqn(DQN_PATH)

    print('\nLoading ensemble heads...')
    heads = load_ensemble_heads(MODEL_PATH, dqn_model)

    # ── 3. Inference ──────────────────────────────────────────────────────────
    print('\nComputing Q-values for all frames...')
    q_all = extract_q_all(heads, dqn_model, states)      # (N, H, 6)
    print(f'  q_all shape: {q_all.shape}')

    aleatoric, epistemic, total = compute_uncertainty(q_all)
    unc_dict = {'aleatoric': aleatoric, 'epistemic': epistemic, 'total': total}

    print(f'  aleatoric : mean={aleatoric.mean():.4f}  std={aleatoric.std():.4f}'
          f'  max_possible={np.log(6):.4f}')
    print(f'  epistemic : mean={epistemic.mean():.4f}  std={epistemic.std():.4f}')
    print(f'  total     : mean={total.mean():.4f}  std={total.std():.4f}')

    # ── 4. Behavioral signals ─────────────────────────────────────────────────
    print('\nComputing behavioral signals...')
    fut_rew   = future_reward(time_, reward, HORIZON_SEC)
    switch    = compute_switch_indicator(actual_action)
    future_sw = future_window_mean(time_, switch, HORIZON_SEC)

    valid_r = np.isfinite(fut_rew)
    valid_s = np.isfinite(future_sw)
    print(f'  future reward   : mean={fut_rew[valid_r].mean():.4f}'
          f'  valid frames={valid_r.sum():,}')
    print(f'  switch rate     : {switch.mean():.4f}')
    print(f'  future switch   : mean={future_sw[valid_s].mean():.4f}'
          f'  valid frames={valid_s.sum():,}')

    for key, _, label in UNCERTAINTY_CFG:
        r_r, p_r = stats.pearsonr(unc_dict[key][valid_r], fut_rew[valid_r])
        r_s, p_s = stats.pearsonr(unc_dict[key][valid_s], future_sw[valid_s])
        print(f'  {key:<12s}: r(fut_rew)={r_r:+.4f} p={p_r:.2e}  '
              f'r(future_sw)={r_s:+.4f} p={p_s:.2e}')

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    print('\nGenerating figures...')

    lags   = list(np.arange(-6.0, 6.5, 0.5))
    window = 3.0

    plot_uncertainty_timeseries(
        time_, aleatoric, epistemic, total, reward,
        OUT_DIR / 'uncertainty_timeseries.png')

    plot_future_reward_binned(
        unc_dict, fut_rew, HORIZON_SEC,
        OUT_DIR / 'future_reward_binned.png')

    plot_future_reward_lag(
        unc_dict, time_, reward, lags, window,
        OUT_DIR / 'future_reward_lag.png')

    plot_action_switch_binned(
        unc_dict, future_sw, HORIZON_SEC,
        OUT_DIR / 'action_switch_binned.png')

    plot_action_switch_scatter(
        unc_dict, future_sw, HORIZON_SEC,
        OUT_DIR / 'action_switch_scatter.png')

    plot_uncertainty_dist(
        unc_dict, actual_action,
        OUT_DIR / 'uncertainty_dist.png')

    plot_correlation_bar(
        unc_dict, fut_rew, future_sw,
        OUT_DIR / 'correlation_bar.png')

    save_summary_csv(
        unc_dict, fut_rew, future_sw, actual_action,
        OUT_DIR / 'summary_stats.csv')

    print(f'\nDone.  All outputs in: {OUT_DIR}')


if __name__ == '__main__':
    main()
