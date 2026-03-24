"""
compute_glm_regressors.py

GLM regressor matrix 생성.
여러 file_idx를 한 번에 처리해 12개의 MAT 파일과 1개의 CSV 파일로 출력한다.

Pixel novelty (3종):
  u_pixel_ae             AE reconstruction MSE
  u_pixel_vae            VAE reconstruction MSE
  u_pixel_rnd            RND prediction error

Q-level uncertainty (4종):
  u_q_unc_kendall        1 - Kendall's W (rank disagreement, ↑=uncertain)
  u_q_vote_disagree      1 - max_vote / n_heads (↑=uncertain)
  u_q_zscore_std         Z-score std of ensemble top-1 action (↑=uncertain)
  u_q_zscore_human       Z-score std of human-chosen action across heads (↑=uncertain)

HRF 처리:
  SPM canonical HRF (double-gamma, 32s), dt=0.05s fine grid
  event regressors   : impulse at frame time → HRF conv → TR 샘플링 → z-score
  continuous regressors: mean-center over game frames → piecewise-constant → HRF conv → TR 샘플링 → z-score
  TR = 1.0s, game 구간만 처리 (~480 TRs); pre/post rest 구간 제외

출력 MAT 파일 (36개, run × 조합별):
  regressors_{subject}_f{fidx}_{pixel}_{unc}.mat
  R 행 수: ~480 (game TRs only), 열 구조:
    action_direction        HRF-convolved onset event × 2  (2 cols)
    action_fire             (NOOP은 baseline으로 제외; onset = action bout 시작 프레임)
                            direction: RIGHT, LEFT, RIGHT+FIRE, LEFT+FIRE (방향키 눌림)
                            fire:      FIRE, RIGHT+FIRE, LEFT+FIRE (fire 버튼 눌림)
                            RIGHT+FIRE / LEFT+FIRE는 두 regressor 모두에 기여
    reward                  HRF-convolved event (raw magnitude)  (1 col)
    terminal                HRF-convolved event            (1 col)
    frame_diff_mse          HRF-convolved continuous       (1 col)
    u_pixel_{type}          HRF-convolved continuous       (1 col)
    u_q_{type}              HRF-convolved continuous       (1 col)
  모든 regressor는 run별 z-score 정규화됨 (HRF 이후, game TR 기준)
  MAT 메타데이터 (GLM regressor 아님):
    time      TR 시간축 (0, 1, ..., N_tr-1)
    file_idx  npz 파일 인덱스 (scalar)

출력 CSV 파일 (1개, 상관분석용):
  run별로 연결된 비-block 형태의 모든 regressors
"""

import sys
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
from tqdm import tqdm

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'variance' / 'pixel'))
sys.path.insert(0, str(_ROOT / 'variance' / 'q-level'))

from utils import robust_torch_load


# ─── Argument parsing ─────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='GLM regressor matrix 생성')

    # Data
    p.add_argument('--data_dir', default=str(_ROOT / 'processed_data_frameskip_4'))
    p.add_argument('--subject',  default='sub_1',
                   choices=['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6'])
    p.add_argument('--file_idx', nargs='+', type=int, default=[8, 9, 10],
                   help='npz 파일 인덱스 (0-based), 여러 개 가능 (예: --file_idx 8 9 10)')

    # Model paths
    p.add_argument('--dqn_path',      default=str(_ROOT / 'pretrained' / 'dqn_cnn.pt'),
                   help='Pretrained DQN CNN checkpoint (CNN encoder용)')
    p.add_argument('--ae_path',       default=str(_ROOT / 'trained_models/sub_1_ae.pth'))
    p.add_argument('--vae_path',      default=str(_ROOT / 'trained_models/sub_1_vae.pth'))
    p.add_argument('--rnd_path',      default=str(_ROOT / 'trained_models/sub_1_rnd.pth'))
    p.add_argument('--ensemble_path', default=str(_ROOT / 'trained_models/sub_1_dqn.pth'))

    # Options
    p.add_argument('--batch_size',  type=int, default=256)
    p.add_argument('--device',      default='auto')
    p.add_argument('--out_dir',     default=str(_ROOT / 'mat/sub1_run8910'))

    return p.parse_args()


# ─── Device helper ────────────────────────────────────────────────────────────

def resolve_device(spec: str) -> torch.device:
    if spec == 'auto':
        if torch.backends.mps.is_available():  return torch.device('mps')
        if torch.cuda.is_available():          return torch.device('cuda')
        return torch.device('cpu')
    return torch.device(spec)


# ─── Pixel-novelty model loaders ──────────────────────────────────────────────

def load_ae(path: str, device: torch.device):
    from ae import AE
    ck = robust_torch_load(path)
    m  = AE(latent_dim=ck['latent_dim']).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    print(f'  AE  loaded: {Path(path).name}  '
          f'(epoch={ck["epoch"]}, latent_dim={ck["latent_dim"]})')
    return m


def load_vae(path: str, device: torch.device):
    from vae import VAE
    ck = robust_torch_load(path)
    m  = VAE(latent_dim=ck['latent_dim']).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    print(f'  VAE loaded: {Path(path).name}  '
          f'(epoch={ck["epoch"]}, latent_dim={ck["latent_dim"]}, beta={ck["beta"]})')
    return m


def load_rnd(path: str, device: torch.device):
    from rnd import RND
    ck = robust_torch_load(path)
    m  = RND(feature_dim=ck['feature_dim']).to(device)
    m.predictor.load_state_dict(ck['predictor'])
    m.target.load_state_dict(ck['target'])
    m.eval()
    print(f'  RND loaded: {Path(path).name}  '
          f'(epoch={ck["epoch"]}, feature_dim={ck["feature_dim"]})')
    return m


# ─── Ensemble DQN loader ──────────────────────────────────────────────────────

def _load_heads_sd(path: str) -> dict:
    """
    Ensemble heads state dict 로드.

    robust_torch_load를 사용하므로 EOCD가 없는 구형 모델도 자동 복구된다.
    - 신형 모델: torch.load가 직접 성공 → 복구 경로 미사용
    - 구형 모델: torch.load 실패 → EOCD 복구 후 재로드
    """
    sd = robust_torch_load(path)
    if isinstance(sd, dict):
        return sd
    raise RuntimeError(
        f"Loaded object from {path} is not a dict (type: {type(sd)}). "
        "Expected an OrderedDict (model.heads.state_dict())."
    )


def load_ensemble(path: str, dqn_path: str, device: torch.device):
    from model.dqn import load_pretrained_cnn

    sd = _load_heads_sd(path)

    # head 수 자동 감지: 키 패턴 '0.0.weight', '1.0.weight', ...
    n = 0
    while f'{n}.0.weight' in sd and tuple(sd[f'{n}.0.weight'].shape) == (512, 3136):
        n += 1
    if n == 0:
        raise RuntimeError('Ensemble checkpoint: no valid heads found.')

    cnn = load_pretrained_cnn(dqn_path, freeze=True)

    class _Ens(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn   = cnn
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, 6))
                for _ in range(n)
            ])

        def forward(self, x):
            f = self.cnn(x)
            return torch.stack([h(f) for h in self.heads], dim=1)  # (B, H, 6)

    model = _Ens().to(device)
    for p in model.cnn.parameters():
        p.requires_grad_(False)
    model.heads.load_state_dict({k: v for k, v in sd.items()
                                  if int(k.split('.')[0]) < n})
    model.eval()
    print(f'  Ensemble loaded: {Path(path).name}  ({n} heads)')
    return model


# ─── Regressor computation ────────────────────────────────────────────────────

def batched(arr, batch_size):
    """numpy array를 batch_size 단위로 yield."""
    for i in range(0, len(arr), batch_size):
        yield i, arr[i:i + batch_size]


def compute_pixel_novelty(model, states_uint8: np.ndarray,
                           batch_size: int, device: torch.device) -> np.ndarray:
    """
    states_uint8: (N, 4, 84, 84) uint8
    returns: (N,) float32 novelty scores via model.novelty_score()
    """
    out = []
    with torch.no_grad():
        for _, batch in tqdm(list(batched(states_uint8, batch_size)),
                             desc='    novelty', leave=False):
            x = torch.from_numpy(batch[:, -1:, :, :]).to(device).float() / 255.0
            out.append(model.novelty_score(x).cpu().numpy())
    return np.concatenate(out)


def compute_q_uncertainty(model, states_uint8: np.ndarray, actions: np.ndarray,
                           batch_size: int, device: torch.device) -> dict:
    """
    states_uint8: (N, 4, 84, 84) uint8
    actions:      (N,) int32 – human action indices (argmax of one-hot)
    returns: dict of (N,) arrays for each of 4 uncertainty metrics
    """
    kw, vd, zs, zh = [], [], [], []
    with torch.no_grad():
        for i, batch in tqdm(list(batched(states_uint8, batch_size)),
                             desc='    q-uncertainty', leave=False):
            x = torch.from_numpy(batch).to(device).float() / 255.0
            q_all = model(x)          # (B, H, A)
            B, H, A = q_all.shape

            # shared z-score computation (per-head, per-action)
            q_mu  = q_all.mean(dim=-1, keepdim=True)
            q_sig = q_all.std(dim=-1, keepdim=True).clamp_min(1e-8)
            z_all = (q_all - q_mu) / q_sig  # (B, H, A)

            # 1. unc_kendall = 1 - W
            ranks = torch.argsort(torch.argsort(q_all, dim=-1), dim=-1).float()
            R_j   = ranks.sum(dim=1)
            R_bar = R_j.mean(dim=-1, keepdim=True)
            S     = ((R_j - R_bar) ** 2).sum(dim=-1)
            W     = 12 * S / (H * H * (A ** 3 - A))
            kw.append((1.0 - W).cpu().numpy())

            # 2. vote disagreement
            votes     = q_all.argmax(dim=-1)
            max_votes = torch.zeros(B, device=device)
            for a in range(A):
                max_votes = torch.max(max_votes, (votes == a).sum(dim=-1).float())
            vd.append((1.0 - max_votes / H).cpu().numpy())

            # 3. Z-score std of ensemble top-1 action
            mean_q = q_all.mean(dim=1)
            top1_a = mean_q.argmax(dim=-1)
            t1_idx = top1_a.view(B, 1, 1).expand(B, H, 1)
            zs.append(z_all.gather(-1, t1_idx).squeeze(-1).std(dim=-1).cpu().numpy())

            # 4. Z-score std of human-chosen action
            ha_idx = torch.from_numpy(actions[i:i + B]).to(device).long()
            ha_idx = ha_idx.view(B, 1, 1).expand(B, H, 1)
            zh.append(z_all.gather(-1, ha_idx).squeeze(-1).std(dim=-1).cpu().numpy())

    return {
        'u_q_unc_kendall':   np.concatenate(kw),
        'u_q_vote_disagree': np.concatenate(vd),
        'u_q_zscore_std':    np.concatenate(zs),
        'u_q_zscore_human':  np.concatenate(zh),
    }


# ─── HRF helpers ──────────────────────────────────────────────────────────────

def spm_hrf(dt: float = 0.05, t_end: float = 32.0) -> np.ndarray:
    """
    SPM canonical HRF (double-gamma, canonical only — no derivative).
    SPM12 default parameters: peak at ~5s, undershoot at ~15s.
    """
    from scipy.stats import gamma as _gamma
    t = np.arange(0.0, t_end, dt)
    h = (_gamma.pdf(t, 6.0, scale=1.0)
         - _gamma.pdf(t, 16.0, scale=1.0) / 6.0)
    h /= h.max()          # peak-normalize (SPM convention)
    return h.astype(np.float64)


def build_fine_regressor(t_frames: np.ndarray, values: np.ndarray,
                          fine_t: np.ndarray, kind: str) -> np.ndarray:
    """
    Construct regressor signal on a fine time grid.

    kind='continuous': piecewise-constant; values[i] holds over [t_frames[i], t_frames[i+1]).
    kind='event'     : delta impulse at t_frames[i] scaled by values[i].
    """
    sig = np.zeros(len(fine_t), dtype=np.float64)
    if kind == 'continuous':
        for i in range(len(t_frames) - 1):
            mask = (fine_t >= t_frames[i]) & (fine_t < t_frames[i + 1])
            sig[mask] = values[i]
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
    indices  = np.searchsorted(fine_t, tr_times, side='left')
    indices  = np.clip(indices, 0, len(conv_sig) - 1)
    return conv_sig[indices].astype(np.float64), tr_times


def _zscore(v: np.ndarray) -> np.ndarray:
    """Z-score a 1-D array. std가 0이면 zeros 반환."""
    std = v.std()
    if std < 1e-8:
        return np.zeros_like(v, dtype=np.float64)
    return ((v - v.mean()) / std).astype(np.float64)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args         = get_args()
    device       = resolve_device(args.device)
    file_indices = args.file_idx

    # action 분류 (독립 버튼 개념):
    #   direction(왼손): RIGHT(2), LEFT(3), RIGHT+FIRE(4), LEFT+FIRE(5)
    #   fire(오른손):    FIRE(1),  RIGHT+FIRE(4), LEFT+FIRE(5)
    #   NOOP (idx=0) 은 baseline 제외
    # ACTION index: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHT+FIRE, 5=LEFT+FIRE
    ACTION_ONSET_REGS    = ['action_direction', 'action_fire']
    PIXEL_TYPES  = ['ae', 'vae', 'rnd']
    UNCT_TYPES   = ['unc_kendall', 'vote_disagree', 'zscore_std', 'zscore_human']
    U_PIXEL_KEYS = [f'u_pixel_{pt}' for pt in PIXEL_TYPES]
    U_QUNC_KEYS  = [f'u_q_{ut}'     for ut in UNCT_TYPES]

    # base regressors (no pixel/uncertainty — added per-combination)
    BASE_REGS    = ACTION_ONSET_REGS
    # HRF parameters
    HRF_DT       = 0.05    # fine grid resolution (s)
    HRF_LEN      = 32.0    # HRF kernel length (s)
    TR           = 1.0     # fMRI TR (s)

    # regressor type classification (for HRF pipeline)
    CONTINUOUS_REGS = U_PIXEL_KEYS + U_QUNC_KEYS
    EVENT_REGS      = ACTION_ONSET_REGS
    out_dir = Path(args.out_dir) if args.out_dir else Path('.')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Device: {device}')
    print(f'File indices: {file_indices}')

    # ── 1. Validate npz files ─────────────────────────────────────────────────
    npz_files = sorted(glob.glob(
        str(Path(args.data_dir) / args.subject / '*.npz')))
    if not npz_files:
        raise FileNotFoundError(f'No npz files in {args.data_dir}/{args.subject}')
    for fidx in file_indices:
        if fidx < 0 or fidx >= len(npz_files):
            raise ValueError(f'file_idx {fidx} out of range [0, {len(npz_files)-1}]')

    # ── 2. Load models once ───────────────────────────────────────────────────
    print('\nLoading pixel-novelty models...')
    ae_model  = load_ae(args.ae_path,   device)
    vae_model = load_vae(args.vae_path, device)
    rnd_model = load_rnd(args.rnd_path, device)

    print('\nLoading ensemble DQN...')
    ens_model = load_ensemble(args.ensemble_path, args.dqn_path, device)

    # ── 3. Process each run: compute raw frame-level regressors ──────────────
    per_run = []   # list of (fidx, N, run_dict)
    for fidx in file_indices:
        npz_path = npz_files[fidx]
        print(f'\n{"="*60}')
        print(f'Processing file_idx={fidx}: {Path(npz_path).name}')

        data   = np.load(npz_path)
        time_  = data['time'].astype(np.float64)
        states = data['state']                              # (N, 4, 84, 84) uint8
        acts   = data['action']                             # (N, 6) one-hot
        N      = len(time_)
        print(f'  Frames: {N:,}  |  time: {time_[0]:.2f}s – {time_[-1]:.2f}s')

        action_idx = acts.argmax(axis=1).astype(np.int32)

        # 버튼별 binary 시계열
        # direction(왼손): RIGHT(2), LEFT(3), RIGHT+FIRE(4), LEFT+FIRE(5)
        # fire(오른손):    FIRE(1),  RIGHT+FIRE(4), LEFT+FIRE(5)
        # NOOP(0) = baseline
        dir_pressed  = np.isin(action_idx, [2, 3, 4, 5]).astype(np.float64)
        fire_pressed = np.isin(action_idx, [1, 4, 5]).astype(np.float64)

        # 버튼별 onset: 각 버튼이 0→1로 눌리는 순간만 (FIRE→RIGHT+FIRE 시 fire onset 없음)
        dir_onset        = np.zeros(N, dtype=np.float64)
        fire_onset       = np.zeros(N, dtype=np.float64)
        dir_onset[0]     = dir_pressed[0]
        fire_onset[0]    = fire_pressed[0]
        dir_onset[1:]    = np.maximum(dir_pressed[1:]  - dir_pressed[:-1],  0)
        fire_onset[1:]   = np.maximum(fire_pressed[1:] - fire_pressed[:-1], 0)

        print('  Computing pixel novelty...')
        u_ae  = compute_pixel_novelty(ae_model,  states, args.batch_size, device).astype(np.float64)
        u_vae = compute_pixel_novelty(vae_model, states, args.batch_size, device).astype(np.float64)
        u_rnd = compute_pixel_novelty(rnd_model, states, args.batch_size, device).astype(np.float64)

        print('  Computing Q-level uncertainty...')
        q_metrics = compute_q_uncertainty(ens_model, states, action_idx,
                                          args.batch_size, device)

        rd = {
            'time':              time_,
            'action_direction':  dir_onset,
            'action_fire':       fire_onset,
            'u_pixel_ae':        u_ae,
            'u_pixel_vae':       u_vae,
            'u_pixel_rnd':       u_rnd,
            'u_q_unc_kendall':   q_metrics['u_q_unc_kendall'].astype(np.float64),
            'u_q_vote_disagree': q_metrics['u_q_vote_disagree'].astype(np.float64),
            'u_q_zscore_std':    q_metrics['u_q_zscore_std'].astype(np.float64),
            'u_q_zscore_human':  q_metrics['u_q_zscore_human'].astype(np.float64),
        }

        per_run.append((fidx, N, rd))

    # ── 3.5 Time interval diagnostics (frame-level) ──────────────────────────
    print('\n' + '='*60)
    print('Time interval diagnostics (frame-to-frame dt):')
    for fidx, N, rd in per_run:
        dt = np.diff(rd['time'])
        print(f'  run file_idx={fidx}  N={N:,}')
        print(f'    time range : {rd["time"][0]:.4f}s – {rd["time"][-1]:.4f}s')
        print(f'    dt  mean   : {dt.mean():.6f}s')
        print(f'    dt  std    : {dt.std():.6f}s')
        print(f'    dt  min    : {dt.min():.6f}s')
        print(f'    dt  max    : {dt.max():.6f}s')
        # histogram of dt rounded to 3 decimals
        vals, counts = np.unique(dt.round(3), return_counts=True)
        top_n = min(5, len(vals))
        top_idx = np.argsort(-counts)[:top_n]
        top_str = '  '.join(f'{vals[i]:.3f}s×{counts[i]}' for i in top_idx)
        print(f'    dt  top-{top_n} : {top_str}')
    print('='*60)

    # ── 3.7 HRF convolution + TR resampling (game-only, per run) ─────────────
    # continuous: mean-center over game frames → piecewise-const → HRF conv → TR resample → z-score
    # event:      impulse at frame time (no mean-center) → HRF conv → TR resample → z-score
    print('\nApplying HRF convolution and TR resampling...')
    hrf = spm_hrf(dt=HRF_DT, t_end=HRF_LEN)

    per_run_tr = []
    for fidx, N, rd in per_run:
        t       = rd['time'] - rd['time'][0]   # zero-aligned game time (s)
        run_dur = float(t[-1])
        fine_t  = np.arange(0.0, run_dur + HRF_LEN + HRF_DT, HRF_DT)

        rd_tr = {}
        for key in CONTINUOUS_REGS:
            vals          = rd[key] - rd[key].mean()   # mean-center over game frames
            sig           = build_fine_regressor(t, vals, fine_t, kind='continuous')
            conv          = convolve_hrf(sig, hrf)
            sampled, tr_t = resample_to_tr(conv, fine_t, TR, run_dur)
            rd_tr[key]    = _zscore(sampled)

        for key in EVENT_REGS:
            sig           = build_fine_regressor(t, rd[key], fine_t, kind='event')
            conv          = convolve_hrf(sig, hrf)
            sampled, tr_t = resample_to_tr(conv, fine_t, TR, run_dur)
            rd_tr[key]    = sampled   # no z-score; SPM handles scaling

        rd_tr['time_tr'] = tr_t
        N_tr = len(tr_t)
        print(f'  file_idx={fidx}: {N:,} frames → {N_tr} game TRs  (run_dur={run_dur:.1f}s)')
        per_run_tr.append((fidx, N_tr, rd_tr))

    # ── 4. Build per-run MAT files (36 total: n_runs × n_pixel × n_unc) ─────────
    saved_mats = []
    for fidx, N_tr, rd_tr in per_run_tr:
        for pt in PIXEL_TYPES:
            for ut in UNCT_TYPES:
                pixel_key = f'u_pixel_{pt}'
                unc_key   = f'u_q_{ut}'

                col_names  = []
                col_arrays = []
                for reg in BASE_REGS + [pixel_key, unc_key]:
                    col_names.append(reg)
                    col_arrays.append(rd_tr[reg])

                R = np.column_stack(col_arrays)

                mat_path = out_dir / f'regressors_{args.subject}_f{fidx}_{pt}_{ut}.mat'
                scipy.io.savemat(str(mat_path), {
                    'R':        R,
                    'names':    np.array(col_names, dtype=object),
                    'time':     rd_tr['time_tr'],
                    'file_idx': np.int32(fidx),
                })
                saved_mats.append(mat_path)
                print(f'Saved → {mat_path}  ({N_tr} TRs × {R.shape[1]} cols)')

    # ── 5. Save CSV (concatenated, non-block, all regressors) ─────────────────
    csv_rows = []
    for fidx, N, rd in per_run_tr:
        row_dict = {'run_idx': np.full(N, fidx, dtype=np.int32)}
        row_dict['time']   = rd['time_tr']
        for reg in ACTION_ONSET_REGS:
            row_dict[reg] = rd[reg]
        for key in U_PIXEL_KEYS + U_QUNC_KEYS:
            row_dict[key] = rd[key]
        csv_rows.append(pd.DataFrame(row_dict))

    csv_df   = pd.concat(csv_rows, ignore_index=True)
    fidx_str = '_'.join(str(i) for i in file_indices)
    csv_path = out_dir / f'regressors_{args.subject}_f{fidx_str}.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f'\nSaved CSV → {csv_path}')

    # ── 5.5 Regressor time-series visualization ───────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    REG_ORDER = BASE_REGS + U_PIXEL_KEYS + U_QUNC_KEYS
    SHORT = {
        'action_direction':  'dir_onset',
        'action_fire':       'fire_onset',
        'u_pixel_ae':        'pix_ae',
        'u_pixel_vae':       'pix_vae',
        'u_pixel_rnd':       'pix_rnd',
        'u_q_unc_kendall':   'q_kendall',
        'u_q_vote_disagree': 'q_vote',
        'u_q_zscore_std':    'q_zstd',
        'u_q_zscore_human':  'q_zhuman',
    }

    print('\nSaving regressor time-series plots...')
    for fidx, N_tr, rd_tr in per_run_tr:
        n_regs = len(REG_ORDER)
        fig, axes = plt.subplots(n_regs, 1, figsize=(16, n_regs * 1.6), sharex=True)
        t_axis = rd_tr['time_tr']
        for ax, key in zip(axes, REG_ORDER):
            ax.plot(t_axis, rd_tr[key], linewidth=0.7, color='steelblue')
            ax.set_ylabel(SHORT[key], fontsize=7, rotation=0, labelpad=60, va='center')
            ax.tick_params(labelsize=6)
            ax.spines[['top', 'right']].set_visible(False)
        axes[-1].set_xlabel('Time (s)', fontsize=8)
        fig.suptitle(f'{args.subject}  |  run f{fidx} — Regressors (post-HRF, pre-MAT)',
                     fontsize=10, y=1.001)
        fig.tight_layout()
        ts_path = out_dir / f'timeseries_{args.subject}_f{fidx}.png'
        fig.savefig(str(ts_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved → {ts_path}')

    # ── 6. Correlation matrix visualization ───────────────────────────────────
    labels = [SHORT[k] for k in REG_ORDER]
    n_regs = len(REG_ORDER)

    print('\nSaving regressor correlation matrices...')
    for fidx, N_tr, rd_tr in per_run_tr:
        mat  = np.column_stack([rd_tr[k] for k in REG_ORDER])
        corr = np.corrcoef(mat.T)

        sz  = max(8, n_regs * 0.7 + 1)
        fig, ax = plt.subplots(figsize=(sz, sz))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n_regs))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_regs))
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(n_regs):
            for j in range(n_regs):
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                        fontsize=6,
                        color='white' if abs(corr[i, j]) > 0.6 else 'black')
        ax.set_title(f'{args.subject}  |  run f{fidx} — Regressor Correlation Matrix')
        fig.tight_layout()
        png_path = out_dir / f'corr_{args.subject}_f{fidx}.png'
        fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved → {png_path}')

    # ── 7. Summary ────────────────────────────────────────────────────────────
    n_runs = len(file_indices)
    n_cols = len(BASE_REGS) + 2   # base regs + pixel + unc (no run constant)
    n_trs  = [N for _, N, _ in per_run_tr]
    print(f'\nTotal MAT files : {len(saved_mats)}  ({n_runs} runs × {len(PIXEL_TYPES)} pixel × {len(UNCT_TYPES)} unc)')
    print(f'  Per-run TRs     : {n_trs}')
    print(f'  Cols per MAT    : {n_cols}  ({len(BASE_REGS)} base + 1 pixel + 1 unc)')
    print(f'  HRF             : SPM canonical, {HRF_LEN:.0f}s, dt={HRF_DT}s → TR={TR}s')
    print(f'  Continuous regs : mean-centered over game frames before HRF conv')
    print(f'  Z-score         : game-only TR basis, per run')
    print(f'\nMAT file naming: regressors_{{subject}}_f{{fidx}}_{{pixel}}_{{unc}}.mat')


if __name__ == '__main__':
    main()
