"""
glm_utils.py — Shared utilities for compute_glm_regressors_*.py

Functions shared across all GLM regressor scripts:
  - Pixel feature extraction and PCA fitting/loading
  - DQN model loading and output extraction
  - HRF pipeline (spm_hrf, build_fine_regressor, convolve_hrf, resample_to_tr)
  - Z-score normalization
"""

import glob
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent


# ── Pixel features ────────────────────────────────────────────────────────────

def extract_pixel_features(states_uint8: np.ndarray) -> np.ndarray:
    """
    Full 4-frame stack flattened to pixel feature vector.

    states_uint8 : (N, 4, 84, 84) uint8
    returns      : (N, 28224) float32, normalized to [0, 1]
    """
    flat = states_uint8.reshape(len(states_uint8), -1)
    return flat.astype(np.float32) / 255.0


def collect_pixels_for_pca(data_dir: str, subjects: list[str],
                            file_indices: list[int]) -> np.ndarray:
    """Collect pixel features from multiple subjects/runs for PCA fitting."""
    all_pixels = []
    for subj in subjects:
        npz_files = sorted(glob.glob(str(Path(data_dir) / subj / '*.npz')))
        if not npz_files:
            print(f'  WARNING: {subj}: npz 없음, PCA fitting에서 제외')
            continue
        for fidx in file_indices:
            if fidx >= len(npz_files):
                print(f'  WARNING: {subj} file_idx={fidx}: 파일 없음, 제외')
                continue
            data   = np.load(npz_files[fidx])
            pixels = extract_pixel_features(data['state'])
            all_pixels.append(pixels)
            print(f'    {subj} f{fidx}: {pixels.shape}')
    if not all_pixels:
        raise RuntimeError('pixel features 수집 실패: 유효한 npz 파일이 없습니다.')
    result = np.concatenate(all_pixels, axis=0)
    print(f'  -> 전체 pixel features: {result.shape}  ({result.nbytes / 1e9:.2f} GB)')
    return result


# ── PCA fitting / loading ─────────────────────────────────────────────────────

def fit_pca(pixels_all: np.ndarray | None, n_components: int,
            cache_path: str | None = None):
    """
    Fit pixel PCA or load cached basis.

    pixels_all : (N, 28224) — required when cache absent.
    cache_path : .npz path; load if exists, fit+save otherwise.
    returns    : fitted sklearn PCA object.
    """
    from sklearn.decomposition import PCA

    if cache_path is not None and Path(cache_path).exists():
        print(f'  공유 PCA basis 로드: {cache_path}')
        cached   = np.load(cache_path, allow_pickle=True)
        cached_k = int(cached['components'].shape[0])
        cached_d = int(cached['n_features_in'])
        expected_d = 4 * 84 * 84
        if cached_k != n_components:
            raise ValueError(
                f'PCA basis mismatch: cache has {cached_k} components '
                f'but --pca_dim={n_components}. Delete cache and refit.')
        if cached_d != expected_d:
            raise ValueError(
                f'PCA basis mismatch: cache input dim={cached_d}, '
                f'expected {expected_d} (4x84x84). Delete cache and refit.')
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
        pca.components_               = cached['components']
        pca.explained_variance_       = cached['explained_variance']
        pca.explained_variance_ratio_ = cached['explained_variance_ratio']
        pca.mean_                     = cached['mean']
        pca.n_components_             = n_components
        pca.n_features_in_            = cached_d
        print(f'  -> {n_components} components, input_dim={cached_d}, '
              f'누적 EVR={pca.explained_variance_ratio_.cumsum()[-1]:.3f}')
        return pca

    if pixels_all is None:
        raise ValueError(f'pca_basis_path ({cache_path}) 가 없고 pixels_all도 None: fit 불가.')

    print(f'  PCA fit: input={pixels_all.shape}, n_components={n_components}')
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    pca.fit(pixels_all)
    evr = pca.explained_variance_ratio_
    print(f'  EVR per PC  : {np.round(evr, 4)}')
    print(f'  누적 EVR    : {evr.cumsum()[-1]:.4f}')

    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cp,
                 components=pca.components_,
                 explained_variance=pca.explained_variance_,
                 explained_variance_ratio=pca.explained_variance_ratio_,
                 mean=pca.mean_,
                 n_features_in=np.int64(pca.n_features_in_))
        print(f'  공유 PCA basis 저장 -> {cp}')
    return pca


def fit_latent_pca(features: np.ndarray | None, n_components: int,
                   cache_path: str | None = None, label: str = 'latent'):
    """
    Fit PCA on latent features (DQN fc3, BC/CQL/Ensemble heads), with optional cache.

    features   : (N, D) — required when cache absent.
    cache_path : .npz path; load if exists, fit+save otherwise.
    label      : display name for log messages.
    returns    : fitted sklearn PCA object.
    """
    from sklearn.decomposition import PCA

    if cache_path is not None and Path(cache_path).exists():
        print(f'  {label} PCA basis 로드: {cache_path}')
        cached   = np.load(cache_path, allow_pickle=True)
        cached_k = int(cached['components'].shape[0])
        if cached_k != n_components:
            raise ValueError(
                f'{label} PCA basis mismatch: cache has {cached_k} components '
                f'but --pca_dim={n_components}. Delete cache and refit.')
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
        pca.components_               = cached['components']
        pca.explained_variance_       = cached['explained_variance']
        pca.explained_variance_ratio_ = cached['explained_variance_ratio']
        pca.mean_                     = cached['mean']
        pca.n_components_             = n_components
        pca.n_features_in_            = int(cached['n_features_in'])
        print(f'  -> {n_components} components, input_dim={pca.n_features_in_}, '
              f'누적 EVR={pca.explained_variance_ratio_.cumsum()[-1]:.3f}')
        return pca

    if features is None:
        raise ValueError(f'{label} PCA cache ({cache_path}) 없고 features도 None: fit 불가.')

    print(f'  {label} PCA fit: input={features.shape}, n_components={n_components}')
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    pca.fit(features)
    evr = pca.explained_variance_ratio_
    print(f'  EVR per PC  : {np.round(evr, 4)}')
    print(f'  누적 EVR    : {evr.cumsum()[-1]:.4f}')

    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cp,
                 components=pca.components_,
                 explained_variance=pca.explained_variance_,
                 explained_variance_ratio=pca.explained_variance_ratio_,
                 mean=pca.mean_,
                 n_features_in=np.int64(pca.n_features_in_))
        print(f'  {label} PCA basis 저장 -> {cp}')
    return pca


# ── DQN loading / extraction ─────────────────────────────────────────────────

def load_dqn(dqn_path: str):
    """
    Load DQN (conv + fc3 + fc_out) from checkpoint. Returns model in eval mode on CPU.
    Supports checkpoints storing state_dict directly or under 'policy_net'.
    """
    import sys
    import torch
    sys.path.insert(0, str(_ROOT))
    from model.dqn import DQN

    model = DQN(action_dim=6)
    try:
        from utils import robust_torch_load
        ckpt = robust_torch_load(dqn_path, map_location='cpu')
    except Exception:
        ckpt = torch.load(dqn_path, map_location='cpu')

    state_dict = ckpt.get('policy_net', ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'  DQN loaded: {dqn_path}')
    return model


def extract_dqn_outputs(dqn_model, states_uint8: np.ndarray,
                        batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract DQN Q-values and fc3 latent (512-dim).

    states_uint8 : (N, 4, 84, 84) uint8
    returns      : q_values (N, 6), latent (N, 512)
    """
    import torch
    q_list, lat_list = [], []

    with torch.no_grad():
        for start in range(0, len(states_uint8), batch_size):
            x = torch.from_numpy(
                states_uint8[start:start + batch_size]
            ).float() / 255.0
            h = torch.relu(dqn_model.conv1(x))
            h = torch.relu(dqn_model.conv2(h))
            h = torch.relu(dqn_model.conv3(h))
            h = h.view(h.size(0), -1)
            lat = torch.relu(dqn_model.fc3(h))
            q   = dqn_model.fc_out(lat)
            q_list.append(q.numpy())
            lat_list.append(lat.numpy())

    q_values = np.concatenate(q_list,   axis=0).astype(np.float64)
    latents  = np.concatenate(lat_list, axis=0).astype(np.float32)
    return q_values, latents


def collect_dqn_latents(data_dir: str, subjects: list[str],
                         file_indices: list[int], dqn_model) -> np.ndarray:
    """Collect DQN fc3 latent features from multiple subjects/runs for PCA fitting."""
    all_feats = []
    for subj in subjects:
        npz_files = sorted(glob.glob(str(Path(data_dir) / subj / '*.npz')))
        if not npz_files:
            print(f'  WARNING: {subj}: npz 없음, DQN PCA fitting에서 제외')
            continue
        for fidx in file_indices:
            if fidx >= len(npz_files):
                continue
            data   = np.load(npz_files[fidx])
            _, lat = extract_dqn_outputs(dqn_model, data['state'])
            all_feats.append(lat)
            print(f'    {subj} f{fidx}: DQN latent {lat.shape}')
    if not all_feats:
        raise RuntimeError('DQN latent features 수집 실패.')
    result = np.concatenate(all_feats, axis=0)
    print(f'  -> 전체 DQN latent features: {result.shape}')
    return result


# ── HRF pipeline ─────────────────────────────────────────────────────────────

def spm_hrf(dt: float = 0.05, t_end: float = 32.0) -> np.ndarray:
    """SPM canonical HRF (double-gamma). Peak ~5s, undershoot ~15s."""
    from scipy.stats import gamma as _gamma
    t = np.arange(0.0, t_end, dt)
    h = (_gamma.pdf(t, 6.0, scale=1.0)
         - _gamma.pdf(t, 16.0, scale=1.0) / 6.0)
    h /= h.max()
    return h.astype(np.float64)


def build_fine_regressor(t_frames: np.ndarray, values: np.ndarray,
                          fine_t: np.ndarray, kind: str) -> np.ndarray:
    """
    Construct regressor on a fine time grid.

    kind='continuous': piecewise-constant; values[i] holds over [t_frames[i], t_frames[i+1]).
    kind='event'     : delta impulse at t_frames[i] scaled by values[i].
    """
    sig = np.zeros(len(fine_t), dtype=np.float64)
    if kind == 'continuous':
        for i in range(len(t_frames) - 1):
            mask = (fine_t >= t_frames[i]) & (fine_t < t_frames[i + 1])
            sig[mask] = values[i]
        sig[fine_t >= t_frames[-1]] = values[-1]
    elif kind == 'event':
        for i in range(len(t_frames)):
            fi = int(np.searchsorted(fine_t, t_frames[i], side='left'))
            if fi < len(sig):
                sig[fi] += values[i]
    return sig


def convolve_hrf(sig: np.ndarray, hrf: np.ndarray) -> np.ndarray:
    """Full convolution with HRF, truncated back to input length."""
    return np.convolve(sig, hrf, mode='full')[:len(sig)]


def resample_to_tr(conv_sig: np.ndarray, fine_t: np.ndarray,
                   tr: float, run_duration: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample convolved signal at TR grid: 0, TR, 2*TR, ..., up to run_duration.
    Returns (sampled_values, tr_times).
    """
    tr_times = np.arange(0.0, run_duration + 1e-9, tr)
    indices  = np.clip(np.searchsorted(fine_t, tr_times, side='left'), 0, len(conv_sig) - 1)
    return conv_sig[indices].astype(np.float64), tr_times


def _zscore(v: np.ndarray) -> np.ndarray:
    """Z-score a 1-D array. std가 0이면 zeros 반환."""
    std = v.std()
    if std < 1e-8:
        return np.zeros_like(v, dtype=np.float64)
    return ((v - v.mean()) / std).astype(np.float64)


# ── HRF convolution pipeline (complete run) ───────────────────────────────────

def apply_hrf_pipeline(per_run: list, continuous_regs: list[str],
                        event_regs: list[str],
                        hrf_dt: float = 0.05, hrf_len: float = 32.0,
                        tr: float = 1.0) -> list:
    """
    Apply HRF convolution + TR resampling to all runs.

    per_run        : list of (fidx, N, run_dict) from per-run feature extraction.
    continuous_regs: keys to mean-center -> HRF -> z-score.
    event_regs     : keys to impulse -> HRF (no z-score).
    returns        : list of (fidx, N_tr, rd_tr).
    """
    hrf = spm_hrf(dt=hrf_dt, t_end=hrf_len)
    per_run_tr = []

    for fidx, N, rd in per_run:
        t       = rd['time'] - rd['time'][0]
        run_dur = float(t[-1])
        fine_t  = np.arange(0.0, run_dur + hrf_len + hrf_dt, hrf_dt)

        rd_tr = {}
        for key in continuous_regs:
            vals          = rd[key] - rd[key].mean()
            sig           = build_fine_regressor(t, vals, fine_t, kind='continuous')
            conv          = convolve_hrf(sig, hrf)
            sampled, tr_t = resample_to_tr(conv, fine_t, tr, run_dur)
            rd_tr[key]    = _zscore(sampled)

        for key in event_regs:
            sig           = build_fine_regressor(t, rd[key], fine_t, kind='event')
            conv          = convolve_hrf(sig, hrf)
            sampled, tr_t = resample_to_tr(conv, fine_t, tr, run_dur)
            rd_tr[key]    = sampled

        rd_tr['time_tr'] = tr_t
        N_tr = len(tr_t)
        print(f'  file_idx={fidx}: {N:,} frames -> {N_tr} game TRs  (run_dur={run_dur:.1f}s)')
        per_run_tr.append((fidx, N_tr, rd_tr))

    return per_run_tr


# ── Motor regressors ──────────────────────────────────────────────────────────

def compute_motor_regressors(acts: np.ndarray) -> dict:
    """
    Compute button-level motor regressors from one-hot action array.

    Direction (left hand): RIGHT(2), LEFT(3), RIGHT+FIRE(4), LEFT+FIRE(5)
    Fire     (right hand): FIRE(1), RIGHT+FIRE(4), LEFT+FIRE(5)
    NOOP(0) = baseline

    returns dict with keys:
        action_direction_onset, action_fire_onset,
        action_direction_hold,  action_fire_hold,
        action_switch
    """
    action_idx   = acts.argmax(axis=1).astype(np.int32)
    N            = len(action_idx)
    dir_pressed  = np.isin(action_idx, [2, 3, 4, 5]).astype(np.float64)
    fire_pressed = np.isin(action_idx, [1, 4, 5]).astype(np.float64)

    dir_onset      = np.zeros(N, dtype=np.float64)
    fire_onset     = np.zeros(N, dtype=np.float64)
    dir_onset[0]   = dir_pressed[0]
    fire_onset[0]  = fire_pressed[0]
    dir_onset[1:]  = np.maximum(dir_pressed[1:]  - dir_pressed[:-1],  0)
    fire_onset[1:] = np.maximum(fire_pressed[1:] - fire_pressed[:-1], 0)

    action_switch = np.zeros(N, dtype=np.float64)
    action_switch[1:] = (action_idx[1:] != action_idx[:-1]).astype(np.float64)

    return {
        'action_idx':             action_idx,
        'action_direction_onset': dir_onset,
        'action_fire_onset':      fire_onset,
        'action_direction_hold':  dir_pressed,
        'action_fire_hold':       fire_pressed,
        'action_switch':          action_switch,
    }


# ── Reward regressors ─────────────────────────────────────────────────────────

def compute_reward_regressors(rews: np.ndarray) -> dict:
    """Compute reward onset event and magnitude regressors."""
    return {
        'reward_pos_onset':     (rews > 0).astype(np.float64),
        'reward_pos_magnitude': np.where(rews > 0, rews, 0.0).astype(np.float64),
    }


# ── MAT / CSV output ──────────────────────────────────────────────────────────

def save_mat(out_dir: Path, subject: str, fidx: int, reg_list: list[str],
             rd_tr: dict, pca_dim: int, suffix: str = '') -> Path:
    """Save one MAT file for a single run."""
    import scipy.io
    col_names  = list(reg_list)
    col_arrays = [rd_tr[r] for r in reg_list]
    R          = np.column_stack(col_arrays)
    fname      = f'regressors_{subject}_f{fidx}{suffix}.mat'
    mat_path   = out_dir / fname
    scipy.io.savemat(str(mat_path), {
        'R':        R,
        'names':    np.array(col_names, dtype=object),
        'time':     rd_tr['time_tr'],
        'file_idx': np.int32(fidx),
        'pca_dim':  np.int32(pca_dim),
    })
    print(f'  Saved -> {mat_path}  ({len(rd_tr["time_tr"])} TRs x {R.shape[1]} cols)')
    return mat_path


def save_csv(out_dir: Path, subject: str, file_indices: list[int],
             per_run_tr: list, all_regs: list[str]) -> Path:
    """Save concatenated CSV of all runs."""
    import pandas as pd
    csv_rows = []
    for fidx, N_tr, rd_tr in per_run_tr:
        row_dict = {'run_idx': np.full(N_tr, fidx, dtype=np.int32),
                    'time':    rd_tr['time_tr']}
        for key in all_regs:
            row_dict[key] = rd_tr[key]
        csv_rows.append(pd.DataFrame(row_dict))
    csv_df   = pd.concat(csv_rows, ignore_index=True)
    fidx_str = '_'.join(str(i) for i in file_indices)
    csv_path = out_dir / f'regressors_{subject}_f{fidx_str}.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f'\nSaved CSV -> {csv_path}')
    return csv_path


# ── Time diagnostics ──────────────────────────────────────────────────────────

def print_time_diagnostics(per_run: list) -> None:
    """Print frame-to-frame dt statistics for each run."""
    print('\n' + '=' * 60)
    print('Time interval diagnostics (frame-to-frame dt):')
    for fidx, N, rd in per_run:
        dt = np.diff(rd['time'])
        print(f'  run file_idx={fidx}  N={N:,}')
        print(f'    time range : {rd["time"][0]:.4f}s - {rd["time"][-1]:.4f}s')
        print(f'    dt  mean   : {dt.mean():.6f}s')
        print(f'    dt  std    : {dt.std():.6f}s')
        print(f'    dt  min    : {dt.min():.6f}s')
        print(f'    dt  max    : {dt.max():.6f}s')
        vals, counts = np.unique(dt.round(3), return_counts=True)
        top_n   = min(5, len(vals))
        top_idx = np.argsort(-counts)[:top_n]
        top_str = '  '.join(f'{vals[i]:.3f}s x{counts[i]}' for i in top_idx)
        print(f'    dt  top-{top_n} : {top_str}')
    print('=' * 60)
