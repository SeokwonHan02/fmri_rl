#!/usr/bin/env python3
"""
RSA (Representational Similarity Analysis): DQN vs BC 표현 유사성 vs fMRI ROI

online DQN (pretrained/dqn_cnn.pt) vs offline BC (trained_models_2/sub_N_bc.pth)
피험자 게임 데이터에서 512-dim 잠재 표현 추출 → TR 집계 → PCA → RDM 생성
fMRI ROI (Harvard-Oxford atlas) RDM과 Spearman RSA + Partial RSA (OLS)

Confound 통제:
  - Temporal RDM  : |i - j|
  - Visual RDM 1  : frame_diff_energy |e_i - e_j|
  - Visual RDM 2  : raw pixel PCA (last frame, 84x84) → 1-Pearson dist  (NEW)

Permutation test:
  - DQN / BC 각각: circular shift → null beta
  - BC-DQN delta : 동일 shift를 양쪽에 동시 적용 → null delta_beta  (NEW)

Lag sweep (NEW):
  - lag in [0, 2, 4, 6] TR (기본값, --lag_trs로 변경 가능)
  - model[:T-lag] vs brain[lag:]  (model이 brain보다 lag TR 앞)
  - lag별 RSA score 라인 플롯 저장

------------------------------------------------------------------------
BOLD 파일 자동 해석:
  BOLD_ROOT/sub-00N/run/runK/*.nii(.gz)  (K = file_idx)
  --subject sub_1  →  sub-001  (sub_N → sub-00N 자동 변환)
  --bold_root 기본값: /Users/seokwon/research/BOLD

------------------------------------------------------------------------
출력 CSV 열:
  subject, file_idx, lag_tr, roi, n_trs, n_pairs, n_voxels,
  rho_dqn, p_dqn, beta_dqn, p_partial_dqn, p_perm_dqn,
  rho_bc,  p_bc,  beta_bc,  p_partial_bc,  p_perm_bc,
  delta_rho (= rho_bc - rho_dqn), delta_beta,
  p_perm_delta  ← paired permutation for BC-DQN difference

사용 예시:
  python rsa_dqn_vs_bc.py \\
    --subject sub_1 \\
    --file_idx 0 1 2 \\
    --bc_path trained_models_2/sub_1_bc.pth \\
    --lag_trs 0 2 4 6 \\
    --out_dir rsa_dqn_bc_results/sub_1
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from scipy.signal import detrend as sp_detrend
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

_ROOT = Path(__file__).parent

# ── ROI 정의 (Harvard-Oxford atlas) ──────────────────────────────────────────
# key: (atlas_type, name_substring) 리스트
# atlas_type: 'cort' = cortical, 'sub' = subcortical
# IPL = Angular + Supramarginal union

ROI_SPECS = {
    'Precuneus':      [('cort', 'Precuneus')],
    'Angular':        [('cort', 'Angular')],
    'Supramarginal':  [('cort', 'Supramarginal')],
    'IPL':            [('cort', 'Angular'), ('cort', 'Supramarginal')],
    'SPL':            [('cort', 'Superior Parietal')],
    'vmPFC':          [('cort', 'Frontal Medial')],
    'Caudate':        [('sub', 'Caudate')],
    'Putamen':        [('sub', 'Putamen')],
    'Accumbens':      [('sub', 'Accumbens')],
    'Intracalcarine': [('cort', 'Intracalcarine')],
    'OccipitalPole':  [('cort', 'Occipital Pole')],
}


# ── Argument parsing ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='RSA: DQN vs BC representation similarity with fMRI ROIs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Game data
    p.add_argument('--data_dir', default=str(_ROOT / 'processed_data_frameskip_4'))
    p.add_argument('--subject',  default='sub_2',
                   choices=['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6'])
    p.add_argument('--file_idx', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   help='game npz 파일 인덱스 목록. BOLD/sub-00N/run/runK/ 와 1:1 대응.')

    # fMRI data — 경로는 subject + file_idx에서 자동 결정
    p.add_argument('--bold_root', default='/Users/seokwon/research/BOLD',
                   help='BOLD 루트. {bold_root}/sub-00N/run/runK/*.nii(.gz) 탐색.')
    p.add_argument('--bold_onset_tr', type=int, default=60,
                   help='BOLD game 시작 TR offset (0-based). 기본 60 = 앞 휴식 1분.')
    p.add_argument('--n_game_trs', type=int, default=480,
                   help='game 구간 TR 수. 기본 480 (총 600 - 휴식 60×2).')
    p.add_argument('--no_detrend', action='store_true',
                   help='BOLD detrend 생략')
    p.add_argument('--no_zscore', action='store_true',
                   help='BOLD z-score 생략')

    # Model paths
    p.add_argument('--dqn_path', default=str(_ROOT / 'pretrained' / 'dqn_cnn.pt'))
    p.add_argument('--bc_path',  default=str(_ROOT / 'trained_models_2' / 'sub_2_bc.pth'))

    # PCA
    p.add_argument('--pca_dim', type=int, default=20,
                   help='DQN / BC 잠재 공간 PCA 차원')
    p.add_argument('--pixel_pca_dim', type=int, default=10,
                   help='raw pixel (마지막 프레임 84x84) PCA 차원 (visual confound용)')

    # HRF
    p.add_argument('--apply_hrf', action='store_true',
                   help='모델 피처에 HRF convolution 적용 (default: off)')

    # Lag sweep
    p.add_argument('--lag_trs', nargs='+', type=int, default=[0, 2, 4, 6],
                   help='테스트할 TR lag 목록. model[:T-lag] vs brain[lag:].')

    # RSA
    p.add_argument('--n_perms', type=int, default=1000,
                   help='permutation test 반복 수')
    p.add_argument('--perm_min_shift', type=float, default=0.1,
                   help='circular shift 최소 비율 (TR 수의 비율)')

    # Misc
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--out_dir',    default=str(_ROOT / 'rsa_dqn_bc_results'))

    return p.parse_args()


# ── HRF helpers ───────────────────────────────────────────────────────────────

def _spm_hrf(dt: float = 0.05, t_end: float = 32.0) -> np.ndarray:
    from scipy.stats import gamma as _gamma
    t = np.arange(0.0, t_end, dt)
    h = _gamma.pdf(t, 6.0, scale=1.0) - _gamma.pdf(t, 16.0, scale=1.0) / 6.0
    h /= h.max()
    return h.astype(np.float64)


def apply_hrf_to_features(features_tr: np.ndarray, tr: float = 1.0) -> np.ndarray:
    """
    TR-level feature matrix (T, D)에 SPM canonical HRF convolution 적용.
    각 열을 mean-center → piecewise-const → conv → resample → z-score 처리.
    Returns (T, D) float64.
    """
    HRF_DT  = 0.05
    HRF_LEN = 32.0
    T, D    = features_tr.shape
    hrf     = _spm_hrf(dt=HRF_DT, t_end=HRF_LEN)

    run_dur  = (T - 1) * tr
    fine_t   = np.arange(0.0, run_dur + HRF_LEN + HRF_DT, HRF_DT)
    tr_times = np.arange(0.0, run_dur + 1e-9, tr)
    t_frames = np.arange(T, dtype=np.float64) * tr

    out = np.zeros((len(tr_times), D), dtype=np.float64)
    for d in range(D):
        vals = features_tr[:, d] - features_tr[:, d].mean()
        sig  = np.zeros(len(fine_t))
        for i in range(T - 1):
            mask = (fine_t >= t_frames[i]) & (fine_t < t_frames[i + 1])
            sig[mask] = vals[i]
        sig[fine_t >= t_frames[-1]] = vals[-1]
        conv = np.convolve(sig, hrf, mode='full')[:len(sig)]
        idxs = np.clip(np.searchsorted(fine_t, tr_times, side='left'),
                       0, len(conv) - 1)
        s    = conv[idxs]
        std  = s.std()
        out[:, d] = (s - s.mean()) / (std if std > 1e-8 else 1.0)
    return out


# ── Model loading ─────────────────────────────────────────────────────────────

def load_dqn(dqn_path: str):
    import torch
    sys.path.insert(0, str(_ROOT))
    from model.dqn import DQN

    model = DQN(action_dim=6)
    try:
        from utils import robust_torch_load
        ckpt = robust_torch_load(dqn_path, map_location='cpu')
    except Exception:
        ckpt = torch.load(dqn_path, map_location='cpu')
    model.load_state_dict(ckpt.get('policy_net', ckpt))
    model.eval()
    print(f'  DQN loaded: {dqn_path}')
    return model


def load_bc(bc_path: str):
    import torch
    sys.path.insert(0, str(_ROOT))
    from model.dqn import DQN_CNN
    from model.bc import BehaviorCloning

    if not Path(bc_path).exists():
        raise FileNotFoundError(f'BC model not found: {bc_path}')
    try:
        from utils import robust_torch_load
        state_dict = robust_torch_load(bc_path, map_location='cpu')
    except Exception:
        state_dict = torch.load(bc_path, map_location='cpu')
    model = BehaviorCloning(DQN_CNN(), action_dim=6)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'  BC loaded: {bc_path}')
    return model


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_dqn_latent(dqn_model, states_uint8: np.ndarray,
                       batch_size: int = 512) -> np.ndarray:
    """DQN fc3 512-dim latent 추출. Returns (N, 512) float32."""
    import torch
    N, buf = len(states_uint8), []
    with torch.no_grad():
        for s in range(0, N, batch_size):
            x = torch.from_numpy(states_uint8[s:s + batch_size]).float() / 255.0
            h = torch.relu(dqn_model.conv1(x))
            h = torch.relu(dqn_model.conv2(h))
            h = torch.relu(dqn_model.conv3(h))
            h = h.view(h.size(0), -1)
            buf.append(torch.relu(dqn_model.fc3(h)).numpy())
    return np.concatenate(buf, axis=0).astype(np.float32)


def extract_bc_latent(bc_model, states_uint8: np.ndarray,
                      batch_size: int = 512) -> np.ndarray:
    """
    BC action_head[:2] 512-dim latent 추출.
    action_head = Sequential(Linear(3136,512), ReLU(), Linear(512,6))
    → action_head[:2] = Linear + ReLU → (N, 512)
    """
    import torch
    N, buf = len(states_uint8), []
    with torch.no_grad():
        for s in range(0, N, batch_size):
            x   = torch.from_numpy(states_uint8[s:s + batch_size]).float() / 255.0
            cnn = bc_model.cnn(x)                      # (B, 3136)
            lat = bc_model.action_head[:2](cnn)        # Linear + ReLU → (B, 512)
            buf.append(lat.numpy())
    return np.concatenate(buf, axis=0).astype(np.float32)


def extract_pixel_features(states_uint8: np.ndarray) -> np.ndarray:
    """
    마지막 프레임 (84x84) raw pixel 벡터 추출 (visual confound용).
    states_uint8 : (N, 4, 84, 84) uint8
    returns      : (N, 84*84=7056) float32  ∈ [0, 1]
    """
    last = states_uint8[:, -1, :, :]             # (N, 84, 84)
    return last.reshape(len(last), -1).astype(np.float32) / 255.0


def compute_frame_diff_energy(states_uint8: np.ndarray) -> np.ndarray:
    """연속 프레임 간 MSE (마지막 채널 기준). Returns (N,) float64."""
    last   = states_uint8[:, -1, :, :].astype(np.float32) / 255.0
    energy = np.zeros(len(last), dtype=np.float64)
    energy[1:] = ((last[1:] - last[:-1]) ** 2).mean(axis=(1, 2))
    energy[0]  = energy[1]
    return energy


# ── TR 집계 ───────────────────────────────────────────────────────────────────

def aggregate_to_tr(features_frame: np.ndarray, timestamps: np.ndarray,
                    tr: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    프레임 단위 피처 (N, D) 를 TR 그리드로 평균 집계.

    timestamps : (N,) 초 단위 (절대 시간, 내부에서 0-기준 정렬)
    tr         : TR in seconds

    Returns (features_tr, tr_times):
      features_tr : (T_tr, D) float64
      tr_times    : (T_tr,) 초 (0, TR, 2*TR, ...)
    """
    t_rel    = timestamps - timestamps[0]
    run_dur  = float(t_rel[-1])
    tr_times = np.arange(0.0, run_dur + 1e-9, tr)
    T_tr     = len(tr_times)
    D        = features_frame.shape[1] if features_frame.ndim == 2 else 1

    if features_frame.ndim == 1:
        features_frame = features_frame[:, None]

    features_tr = np.zeros((T_tr, D), dtype=np.float64)
    counts      = np.zeros(T_tr, dtype=np.int32)

    tr_idx = np.clip(
        np.searchsorted(tr_times, t_rel, side='right') - 1,
        0, T_tr - 1,
    )
    for i in range(len(features_frame)):
        features_tr[tr_idx[i]] += features_frame[i]
        counts[tr_idx[i]]      += 1

    mask = counts > 0
    features_tr[mask] /= counts[mask, None]
    # 빈 TR bin은 직전 값으로 forward-fill
    for t in range(1, T_tr):
        if not mask[t]:
            features_tr[t] = features_tr[t - 1]

    return features_tr.astype(np.float64), tr_times


# ── PCA ───────────────────────────────────────────────────────────────────────

def fit_shared_pca(features_list: list[np.ndarray], n_components: int,
                   label: str = '') -> tuple[PCA, list[np.ndarray]]:
    """
    전체 run을 concat하여 PCA fit → 각 run에 transform.

    features_list : list of (T_i, D)
    returns       : (fitted_pca, list of (T_i, n_components))
    """
    all_feats = np.concatenate(features_list, axis=0)
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    pca.fit(all_feats)
    evr = pca.explained_variance_ratio_
    top3 = ', '.join(f'{v:.3f}' for v in evr[:3])
    print(f'  {label} PCA ({n_components}): total_EVR={evr.sum():.3f}  top3=[{top3}]')
    return pca, [pca.transform(f) for f in features_list]


# ── RDM 생성 ──────────────────────────────────────────────────────────────────

def compute_rdm(features: np.ndarray) -> np.ndarray:
    """
    RDM(i,j) = 1 - Pearson correlation.
    features : (T, D)
    returns  : (T, T) symmetric, diagonal=0
    """
    return squareform(pdist(features, metric='correlation'))


def upper_tri_vec(rdm: np.ndarray) -> np.ndarray:
    """strict upper triangle (diagonal 제외) 벡터화."""
    n = rdm.shape[0]
    return rdm[np.triu_indices(n, k=1)]


def make_temporal_rdm(n_trs: int) -> np.ndarray:
    """Temporal confound RDM: |i - j|."""
    idx = np.arange(n_trs)
    return np.abs(idx[:, None] - idx[None, :]).astype(np.float64)


def make_visual_energy_rdm(energy_tr: np.ndarray) -> np.ndarray:
    """
    Visual confound RDM: |energy_i - energy_j| (TR-level scalar).
    energy_tr : (T,)
    returns   : (T, T)
    """
    return np.abs(energy_tr[:, None] - energy_tr[None, :]).astype(np.float64)


# ── fMRI ROI mask 생성 ────────────────────────────────────────────────────────

def load_ho_atlas():
    """
    Harvard-Oxford cortical + subcortical atlas 로드.
    Returns (cort_img, cort_labels, sub_img, sub_labels).
    """
    from nilearn.datasets import fetch_atlas_harvard_oxford
    print('  Loading Harvard-Oxford atlas (cort + sub)...')
    cort = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    sub  = fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    print(f'    cortical: {len(cort["labels"])} labels  '
          f'subcortical: {len(sub["labels"])} labels')
    return cort['maps'], cort['labels'], sub['maps'], sub['labels']


def build_roi_masks(bold_ref_img,
                    cort_img, cort_labels,
                    sub_img,  sub_labels) -> dict[str, np.ndarray]:
    """
    Harvard-Oxford atlas → BOLD 공간으로 resample 후 ROI binary mask 생성.

    bold_ref_img : nibabel Nifti1Image (BOLD의 mean/boldref 볼륨)
    returns      : {roi_name: (X, Y, Z) bool ndarray}
    """
    from nilearn.image import resample_to_img

    cort_res  = resample_to_img(cort_img, bold_ref_img, interpolation='nearest')
    sub_res   = resample_to_img(sub_img,  bold_ref_img, interpolation='nearest')
    cort_data = np.round(cort_res.get_fdata()).astype(np.int32)
    sub_data  = np.round(sub_res.get_fdata()).astype(np.int32)

    roi_masks = {}
    for roi_name, specs in ROI_SPECS.items():
        mask = np.zeros(cort_data.shape, dtype=bool)
        found_any = False
        for atlas_type, substr in specs:
            labels = cort_labels if atlas_type == 'cort' else sub_labels
            data   = cort_data  if atlas_type == 'cort' else sub_data
            hits   = [(i, l) for i, l in enumerate(labels)
                      if substr.lower() in l.lower()]
            if not hits:
                print(f'    WARNING: "{substr}" not found in {atlas_type} atlas')
                continue
            for idx, lab in hits:
                mask |= (data == idx)
                found_any = True
                print(f'    {roi_name}: [{atlas_type}] label[{idx}]="{lab}"')

        n_vox = mask.sum()
        if n_vox == 0 or not found_any:
            print(f'    SKIP "{roi_name}": 0 voxels in BOLD space')
        else:
            roi_masks[roi_name] = mask
            print(f'  ROI "{roi_name}": {n_vox} voxels')

    return roi_masks


# ── BOLD ROI 시계열 추출 ──────────────────────────────────────────────────────

def extract_bold_roi(bold_img, roi_mask: np.ndarray,
                     onset_tr: int = 0, n_trs: int | None = None,
                     detrend: bool = True, zscore: bool = True) -> np.ndarray:
    """
    BOLD 4D NIfTI에서 ROI voxel 시계열 추출.

    bold_img : nibabel Nifti1Image  (X, Y, Z, T_total)
    roi_mask : (X, Y, Z) bool — BOLD 공간 기준
    onset_tr : game 시작 TR (0-based)
    n_trs    : game 구간 TR 수 (None=전체)

    Returns (T_game, V_roi) float64
    """
    data    = bold_img.get_fdata()       # (X, Y, Z, T_total)
    T_total = data.shape[-1]
    end_tr  = (onset_tr + n_trs) if n_trs is not None else T_total
    end_tr  = min(end_tr, T_total)

    bold_roi = data[roi_mask, onset_tr:end_tr].T.astype(np.float64)  # (T_game, V)

    if detrend:
        bold_roi = sp_detrend(bold_roi, axis=0, type='linear')

    if zscore:
        mu  = bold_roi.mean(axis=0, keepdims=True)
        std = bold_roi.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        bold_roi = (bold_roi - mu) / std

    return bold_roi


# ── RSA 계산 ──────────────────────────────────────────────────────────────────

def rsa_spearman(brain_rdm: np.ndarray,
                 model_rdm: np.ndarray) -> tuple[float, float]:
    """Spearman correlation (상삼각, diagonal 제외). Returns (rho, p)."""
    rho, p = scipy.stats.spearmanr(upper_tri_vec(brain_rdm),
                                   upper_tri_vec(model_rdm))
    return float(rho), float(p)


def rsa_partial(brain_rdm: np.ndarray,
                model_rdm: np.ndarray,
                confound_rdms: list[np.ndarray]) -> tuple[float, float]:
    """
    Partial RSA via OLS (statsmodels).
    RDM_brain ~ beta_model * RDM_model + sum(beta_c * RDM_confound_c)

    모든 predictor는 [0, 1]로 정규화, brain RDM vec는 z-score.
    Returns (beta_model, p_model).
    """
    import statsmodels.api as sm

    def _norm01(v: np.ndarray) -> np.ndarray:
        r = v.max() - v.min()
        return (v - v.min()) / (r if r > 1e-10 else 1.0)

    b_vec = upper_tri_vec(brain_rdm).astype(np.float64)
    m_vec = upper_tri_vec(model_rdm).astype(np.float64)

    X_cols = [_norm01(m_vec)]
    for c in confound_rdms:
        X_cols.append(_norm01(upper_tri_vec(c).astype(np.float64)))

    X = sm.add_constant(np.column_stack(X_cols))
    std_b = b_vec.std()
    y = (b_vec - b_vec.mean()) / (std_b if std_b > 1e-8 else 1.0)

    try:
        res  = sm.OLS(y, X).fit()
        beta = float(res.params[1])
        p    = float(res.pvalues[1])
    except Exception as e:
        print(f'      OLS failed: {e}')
        beta, p = np.nan, np.nan

    return beta, p


# ── Permutation tests ─────────────────────────────────────────────────────────

def _build_shifts(T: int, n_perms: int, min_shift_frac: float,
                  seed: int = 42) -> np.ndarray:
    """
    T 길이 시계열용 circular shift 인덱스 배열 생성.
    Returns (n_perms,) int array of shift amounts.
    """
    min_shift = max(1, int(T * min_shift_frac))
    max_shift = T - min_shift
    rng = np.random.default_rng(seed=seed)
    return rng.integers(min_shift, max_shift, size=n_perms).astype(int)


def permutation_test(brain_rdm: np.ndarray,
                     model_feats_tr: np.ndarray,
                     confound_rdms: list[np.ndarray],
                     observed_score: float,
                     n_perms: int = 1000,
                     min_shift_frac: float = 0.1) -> tuple[np.ndarray, float]:
    """
    Model feature 시계열을 circular shift하여 null distribution 생성.

    model_feats_tr : (T, D_pca) — PCA scores
    observed_score : 실제 partial beta
    returns        : (null_scores, p_value)  [one-tailed: P(null >= observed)]
    """
    T       = model_feats_tr.shape[0]
    shifts  = _build_shifts(T, n_perms, min_shift_frac)
    null_scores = np.zeros(n_perms, dtype=np.float64)

    for i, k in enumerate(shifts):
        shifted  = np.roll(model_feats_tr, int(k), axis=0)
        rdm_sh   = compute_rdm(shifted)
        score, _ = rsa_partial(brain_rdm, rdm_sh, confound_rdms)
        null_scores[i] = score if not np.isnan(score) else 0.0

    p_val = float((null_scores >= observed_score).mean())
    return null_scores, p_val


def permutation_test_delta(brain_rdm: np.ndarray,
                           dqn_feats_tr: np.ndarray,
                           bc_feats_tr: np.ndarray,
                           confound_rdms: list[np.ndarray],
                           observed_delta: float,
                           n_perms: int = 1000,
                           min_shift_frac: float = 0.1) -> tuple[np.ndarray, float]:
    """
    BC - DQN delta에 대한 paired circular-shift permutation test.

    동일한 shift k를 DQN·BC 양쪽에 동시 적용 → 같은 temporal autocorrelation
    구조를 유지하면서 brain-model alignment만 파괴.

    null_delta[i] = beta_bc_shifted - beta_dqn_shifted
    p_value (one-tailed): P(null_delta >= observed_delta)

    observed_delta : beta_bc - beta_dqn (관측값)
    returns        : (null_deltas, p_value)
    """
    T       = dqn_feats_tr.shape[0]
    shifts  = _build_shifts(T, n_perms, min_shift_frac)
    null_deltas = np.zeros(n_perms, dtype=np.float64)

    for i, k in enumerate(shifts):
        dqn_sh      = np.roll(dqn_feats_tr, int(k), axis=0)
        bc_sh       = np.roll(bc_feats_tr,  int(k), axis=0)
        beta_dqn, _ = rsa_partial(brain_rdm, compute_rdm(dqn_sh), confound_rdms)
        beta_bc,  _ = rsa_partial(brain_rdm, compute_rdm(bc_sh),  confound_rdms)
        d = beta_bc - beta_dqn
        null_deltas[i] = d if not np.isnan(d) else 0.0

    p_val = float((null_deltas >= observed_delta).mean())
    return null_deltas, p_val


# ── BOLD 파일 자동 탐색 ──────────────────────────────────────────────────────

def subject_to_fmri_id(subject: str) -> str:
    """
    'sub_1' → 'sub-001'  (sub_N → sub-00N, 1자리 숫자는 0-패딩 3자리)
    'sub_10' → 'sub-010' 등
    """
    num = subject.split('_')[-1]
    return f'sub-{int(num):03d}'


def resolve_bold_paths(bold_root: str, subject: str,
                       file_indices: list[int]) -> list[str]:
    """
    {bold_root}/sub-00N/run/runK/ 에서 *desc-preproc_bold.nii(.gz) 탐색.

    subject      : 'sub_1' 형식 → 내부에서 'sub-001'로 변환
    file_indices : 각 run의 인덱스 (K)

    Returns: 각 file_idx에 대응하는 BOLD 파일 경로 목록
    """
    fmri_id   = subject_to_fmri_id(subject)
    root      = Path(bold_root) / fmri_id / 'run'

    if not root.exists():
        raise FileNotFoundError(f'BOLD run 디렉토리 없음: {root}')

    paths = []
    for fidx in file_indices:
        run_dir = root / f'run{fidx}'
        if not run_dir.exists():
            raise FileNotFoundError(f'BOLD run 디렉토리 없음: {run_dir}')

        # .nii 또는 .nii.gz 탐색
        candidates = (sorted(run_dir.glob('*desc-preproc_bold.nii.gz'))
                      + sorted(run_dir.glob('*desc-preproc_bold.nii')))
        if not candidates:
            raise FileNotFoundError(
                f'*desc-preproc_bold.nii(.gz) 파일 없음: {run_dir}')
        if len(candidates) > 1:
            print(f'  WARNING: run{fidx}에 파일이 {len(candidates)}개, '
                  f'첫 번째 사용: {candidates[0].name}')

        paths.append(str(candidates[0]))
        print(f'  file_idx={fidx} → {candidates[0].name}')

    return paths


# ── 시각화 ────────────────────────────────────────────────────────────────────

def save_summary_plot(df_lag0: pd.DataFrame, out_dir: Path, subject: str):
    """lag=0 결과 기반 ROI별 partial beta (DQN vs BC) 막대 그래프 저장."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rois  = df_lag0['roi'].tolist()
    x     = np.arange(len(rois))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(rois) * 1.2 + 2), 5))

    for ax, (col_dqn, col_bc), title in [
        (axes[0], ('beta_dqn', 'beta_bc'),  'Partial RSA (OLS β)'),
        (axes[1], ('rho_dqn',  'rho_bc'),   'Basic RSA (Spearman ρ)'),
    ]:
        ax.bar(x - width / 2, df_lag0[col_dqn], width, label='DQN',
               color='#4878CF', alpha=0.85)
        ax.bar(x + width / 2, df_lag0[col_bc],  width, label='BC',
               color='#E35D2B', alpha=0.85)
        ax.axhline(0, color='#666666', lw=0.8, ls='--')
        ax.set_xticks(x)
        ax.set_xticklabels(rois, rotation=40, ha='right', fontsize=8)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(f'{subject} — {title} (lag=0)', fontsize=10)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / f'rsa_summary_{subject}.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {path}')


def save_lag_plot(df_all: pd.DataFrame, out_dir: Path, subject: str):
    """
    lag별 partial beta (DQN vs BC) 라인 플롯.
    각 ROI에 대해 x축=lag_tr, y축=beta, DQN/BC 두 선.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rois    = sorted(df_all['roi'].unique())
    lags    = sorted(df_all['lag_tr'].unique())
    n_rois  = len(rois)
    n_cols  = min(4, n_rois)
    n_rows  = (n_rois + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3.2),
                             squeeze=False)

    for ax_i, roi in enumerate(rois):
        ax  = axes[ax_i // n_cols][ax_i % n_cols]
        sub = df_all[df_all['roi'] == roi].groupby('lag_tr')[
            ['beta_dqn', 'beta_bc', 'delta_beta']].mean().reset_index()

        ax.plot(sub['lag_tr'], sub['beta_dqn'], 'o-',
                color='#4878CF', lw=1.5, ms=5, label='DQN')
        ax.plot(sub['lag_tr'], sub['beta_bc'],  's-',
                color='#E35D2B', lw=1.5, ms=5, label='BC')
        ax.fill_between(sub['lag_tr'],
                        sub['beta_dqn'], sub['beta_bc'],
                        alpha=0.15, color='gray', label='Δ(BC-DQN)')
        ax.axhline(0, color='#888888', lw=0.6, ls='--')
        ax.set_title(roi, fontsize=9)
        ax.set_xlabel('lag (TR)', fontsize=8)
        ax.set_ylabel('partial β', fontsize=8)
        ax.set_xticks(lags)
        ax.tick_params(labelsize=7)
        if ax_i == 0:
            ax.legend(fontsize=7)

    # 사용하지 않는 subplot 숨김
    for ax_i in range(n_rois, n_rows * n_cols):
        axes[ax_i // n_cols][ax_i % n_cols].set_visible(False)

    fig.suptitle(f'{subject} — RSA by lag (model leads brain)',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    path = out_dir / f'rsa_lag_{subject}.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {path}')


def save_null_dist_plot(null_dqn: np.ndarray, null_bc: np.ndarray,
                        null_delta: np.ndarray,
                        obs_dqn: float, obs_bc: float, obs_delta: float,
                        roi: str, fidx: int, lag: int,
                        out_dir: Path, subject: str):
    """Null distribution 히스토그램 저장 (DQN / BC / delta 3 패널)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    panels = [
        (axes[0], null_dqn,   obs_dqn,   'DQN',        '#4878CF'),
        (axes[1], null_bc,    obs_bc,     'BC',          '#E35D2B'),
        (axes[2], null_delta, obs_delta,  'Δ (BC−DQN)', '#888888'),
    ]
    for ax, null, obs, label, color in panels:
        ax.hist(null, bins=40, color=color, alpha=0.7, edgecolor='none')
        ax.axvline(obs, color='black', lw=1.5, label=f'obs={obs:.4f}')
        p = float((null >= obs).mean())
        ax.set_title(f'{label}: obs={obs:.4f}  p={p:.3f}', fontsize=9)
        ax.set_xlabel('null β', fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle(
        f'{subject} | run f{fidx} | lag={lag} TR | ROI {roi}',
        fontsize=9,
    )
    fig.tight_layout()
    path = out_dir / f'null_{subject}_f{fidx}_lag{lag}_{roi}.png'
    fig.savefig(str(path), dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    file_indices = args.file_idx
    lag_list     = sorted(set(args.lag_trs))
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 65)
    fmri_id = subject_to_fmri_id(args.subject)
    print(f'Subject       : {args.subject}  (fMRI: {fmri_id})')
    print(f'BOLD root     : {args.bold_root}')
    print(f'File indices  : {file_indices}')
    print(f'PCA dim       : {args.pca_dim}  (DQN / BC)')
    print(f'Pixel PCA dim : {args.pixel_pca_dim}  (visual confound)')
    print(f'Apply HRF     : {args.apply_hrf}')
    print(f'Lag TRs       : {lag_list}')
    print(f'n_perms       : {args.n_perms}')
    print(f'DQN path      : {args.dqn_path}')
    print(f'BC path       : {args.bc_path}')
    print(f'Out dir       : {out_dir}')
    print('=' * 65)

    # ── 1. game npz 파일 목록 확인 ────────────────────────────────────────
    npz_files = sorted(glob.glob(
        str(Path(args.data_dir) / args.subject / '*.npz')))
    if not npz_files:
        raise FileNotFoundError(
            f'npz 파일 없음: {args.data_dir}/{args.subject}')
    for idx in file_indices:
        if idx < 0 or idx >= len(npz_files):
            raise ValueError(
                f'file_idx {idx} out of range [0, {len(npz_files)-1}]')

    # ── 2. BOLD 파일 경로 자동 탐색 ──────────────────────────────────────
    print('\nResolving BOLD paths...')
    bold_paths = resolve_bold_paths(args.bold_root, args.subject, file_indices)

    # ── 3. 모델 로드 ──────────────────────────────────────────────────────
    print(f'\n{"=" * 65}')
    print('Loading models...')
    dqn_model = load_dqn(args.dqn_path)
    bc_model  = load_bc(args.bc_path)

    # ── 4. 전체 run에서 frame-level 피처 추출 → TR 집계 ──────────────────
    print(f'\n{"=" * 65}')
    print('Extracting latent features and aggregating to TR grid...')

    dqn_tr_list    = []   # (T_tr_i, 512)
    bc_tr_list     = []
    energy_tr_list = []   # (T_tr_i,)
    pixel_tr_list  = []   # (T_tr_i, 7056)  ← raw pixel (aggregated)
    tr_time_list   = []

    for fidx, bold_path in zip(file_indices, bold_paths):
        npz_path = npz_files[fidx]
        print(f'\n  ── run f{fidx}: {Path(npz_path).name}')
        data   = np.load(npz_path)
        time_  = data['time'].astype(np.float64)
        states = data['state']
        print(f'    frames={len(time_):,}  t=[{time_[0]:.1f}, {time_[-1]:.1f}]s')

        print('    Extracting DQN latent...')
        dqn_lat  = extract_dqn_latent(dqn_model, states, args.batch_size)
        print('    Extracting BC latent...')
        bc_lat   = extract_bc_latent(bc_model,   states, args.batch_size)
        print('    Computing visual features...')
        energy   = compute_frame_diff_energy(states)
        pixels   = extract_pixel_features(states)   # (N, 7056) float32

        dqn_tr,  tr_t = aggregate_to_tr(dqn_lat, time_)
        bc_tr,   _    = aggregate_to_tr(bc_lat,  time_)
        en_tr,   _    = aggregate_to_tr(energy,  time_)
        pix_tr,  _    = aggregate_to_tr(pixels,  time_)  # (T_tr, 7056)

        dqn_tr_list.append(dqn_tr)
        bc_tr_list.append(bc_tr)
        energy_tr_list.append(en_tr[:, 0])
        pixel_tr_list.append(pix_tr)
        tr_time_list.append(tr_t)
        print(f'    T_tr={len(tr_t)}')

    # ── 5. (선택) HRF convolution ─────────────────────────────────────────
    if args.apply_hrf:
        print('\nApplying HRF to model features...')
        dqn_tr_list = [apply_hrf_to_features(f) for f in dqn_tr_list]
        bc_tr_list  = [apply_hrf_to_features(f) for f in bc_tr_list]

    # ── 6. 공유 PCA fit ───────────────────────────────────────────────────
    print(f'\n{"=" * 65}')
    print('Fitting shared PCA (all runs)...')
    _, dqn_pca_list   = fit_shared_pca(dqn_tr_list,   args.pca_dim,       'DQN')
    _, bc_pca_list    = fit_shared_pca(bc_tr_list,    args.pca_dim,       'BC')
    _, pixel_pca_list = fit_shared_pca(pixel_tr_list, args.pixel_pca_dim, 'Pixel')

    # ── 7. ROI mask 생성 (첫 번째 BOLD 기준) ─────────────────────────────
    print(f'\n{"=" * 65}')
    print('Building ROI masks from Harvard-Oxford atlas...')
    import nibabel as nib
    bold_ref = nib.load(bold_paths[0])
    ref_vol  = nib.Nifti1Image(bold_ref.get_fdata()[..., 0], bold_ref.affine)
    cort_img, cort_labels, sub_img, sub_labels = load_ho_atlas()
    roi_masks = build_roi_masks(ref_vol,
                                cort_img, cort_labels,
                                sub_img,  sub_labels)
    if not roi_masks:
        raise RuntimeError('ROI mask가 하나도 생성되지 않았습니다.')

    # ── 8. RSA: run × ROI × lag ───────────────────────────────────────────
    print(f'\n{"=" * 65}')
    print('Running RSA per run × ROI × lag...')

    all_results = []
    null_dir    = out_dir / 'null_dists'
    null_dir.mkdir(exist_ok=True)

    for run_i, (fidx, bold_path) in enumerate(zip(file_indices, bold_paths)):
        dqn_pca_all   = dqn_pca_list[run_i]     # (T_tr, pca_dim)
        bc_pca_all    = bc_pca_list[run_i]
        pixel_pca_all = pixel_pca_list[run_i]   # (T_tr, pixel_pca_dim)
        energy_all    = energy_tr_list[run_i]   # (T_tr,)
        T_tr          = len(energy_all)

        print(f'\n  ── run f{fidx} ({Path(bold_path).name})')

        bold_img   = nib.load(bold_path)
        T_bold     = bold_img.shape[-1]
        # n_game_trs: max 공통 길이 (lag=0 기준)
        n_game_trs = args.n_game_trs if args.n_game_trs is not None else T_tr
        n_game_trs = min(n_game_trs, T_bold - args.bold_onset_tr, T_tr)
        print(f'    T_model={T_tr}  T_bold={T_bold}  '
              f'onset={args.bold_onset_tr}  n_game={n_game_trs}')

        for roi_name, roi_mask in roi_masks.items():
            print(f'    ROI "{roi_name}" ({roi_mask.sum()} voxels)...')

            # bold_roi 한 번만 추출 (최대 길이: n_game_trs)
            try:
                bold_roi_full = extract_bold_roi(
                    bold_img, roi_mask,
                    onset_tr=args.bold_onset_tr,
                    n_trs=n_game_trs,
                    detrend=not args.no_detrend,
                    zscore=not args.no_zscore,
                )
            except Exception as e:
                print(f'      ERROR extracting BOLD: {e}')
                continue

            if bold_roi_full.shape[0] < 10 or bold_roi_full.shape[1] < 2:
                print(f'      SKIP: {bold_roi_full.shape}')
                continue

            for lag in lag_list:
                # lag = L 일 때:
                #   model : TR  0 … T-L-1   (dqn/bc_pca_all[:T-L])
                #   brain : TR  L … T-1     (bold_roi_full[L:T])
                # → model이 brain보다 L TR 앞선 상태를 가정
                eff_T = n_game_trs - lag
                if eff_T < 10:
                    print(f'      lag={lag}: eff_T={eff_T} < 10, skip')
                    continue

                dqn_sc     = dqn_pca_all[:eff_T]
                bc_sc      = bc_pca_all[:eff_T]
                pixel_sc   = pixel_pca_all[:eff_T]
                energy_seg = energy_all[:eff_T]
                bold_lagged = bold_roi_full[lag:lag + eff_T]  # brain[lag:]

                # ── RDMs ──────────────────────────────────────────────────
                rdm_dqn        = compute_rdm(dqn_sc)
                rdm_bc         = compute_rdm(bc_sc)
                rdm_brain      = compute_rdm(bold_lagged)
                rdm_time       = make_temporal_rdm(eff_T)
                rdm_energy     = make_visual_energy_rdm(energy_seg)
                rdm_pixel      = compute_rdm(pixel_sc)   # pixel PCA confound (NEW)
                confounds      = [rdm_time, rdm_energy, rdm_pixel]

                n_pairs = int(eff_T * (eff_T - 1) / 2)

                # ── Basic Spearman RSA ────────────────────────────────────
                rho_dqn, p_dqn = rsa_spearman(rdm_brain, rdm_dqn)
                rho_bc,  p_bc  = rsa_spearman(rdm_brain, rdm_bc)

                # ── Partial RSA (OLS) ─────────────────────────────────────
                beta_dqn, p_partial_dqn = rsa_partial(rdm_brain, rdm_dqn, confounds)
                beta_bc,  p_partial_bc  = rsa_partial(rdm_brain, rdm_bc,  confounds)
                obs_delta               = beta_bc - beta_dqn

                # ── Individual permutation tests ──────────────────────────
                print(f'      lag={lag}  Perm DQN ({args.n_perms})...')
                null_dqn, p_perm_dqn = permutation_test(
                    rdm_brain, dqn_sc, confounds,
                    beta_dqn, args.n_perms, args.perm_min_shift,
                )
                print(f'      lag={lag}  Perm BC  ({args.n_perms})...')
                null_bc, p_perm_bc = permutation_test(
                    rdm_brain, bc_sc, confounds,
                    beta_bc, args.n_perms, args.perm_min_shift,
                )

                # ── Paired delta permutation (NEW) ────────────────────────
                print(f'      lag={lag}  Perm Δ(BC-DQN) ({args.n_perms})...')
                null_delta, p_perm_delta = permutation_test_delta(
                    rdm_brain, dqn_sc, bc_sc, confounds,
                    obs_delta, args.n_perms, args.perm_min_shift,
                )

                # ── Null distribution 그림 ────────────────────────────────
                save_null_dist_plot(
                    null_dqn, null_bc, null_delta,
                    beta_dqn, beta_bc, obs_delta,
                    roi_name, fidx, lag, null_dir, args.subject,
                )

                row = dict(
                    subject=args.subject, file_idx=fidx, lag_tr=lag,
                    roi=roi_name, n_trs=eff_T, n_pairs=n_pairs,
                    n_voxels=int(roi_mask.sum()),
                    # DQN
                    rho_dqn=rho_dqn,   p_dqn=p_dqn,
                    beta_dqn=beta_dqn, p_partial_dqn=p_partial_dqn,
                    p_perm_dqn=p_perm_dqn,
                    # BC
                    rho_bc=rho_bc,     p_bc=p_bc,
                    beta_bc=beta_bc,   p_partial_bc=p_partial_bc,
                    p_perm_bc=p_perm_bc,
                    # Difference
                    delta_rho=rho_bc  - rho_dqn,
                    delta_beta=obs_delta,
                    p_perm_delta=p_perm_delta,   # ← NEW
                )
                all_results.append(row)

                print(f'      lag={lag} DQN: rho={rho_dqn:.3f}  β={beta_dqn:.4f}  '
                      f'p_perm={p_perm_dqn:.3f}')
                print(f'      lag={lag} BC:  rho={rho_bc:.3f}   β={beta_bc:.4f}   '
                      f'p_perm={p_perm_bc:.3f}')
                print(f'      lag={lag} Δ:   rho={rho_bc-rho_dqn:+.3f}  '
                      f'β={obs_delta:+.4f}  p_perm_delta={p_perm_delta:.3f}')

    if not all_results:
        print('\nWARNING: 결과가 없습니다. ROI 매핑 및 BOLD 경로를 확인하세요.')
        return

    # ── 9. 결과 저장 ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = out_dir / f'rsa_results_{args.subject}.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved results → {csv_path}')

    print_cols = ['roi', 'lag_tr', 'rho_dqn', 'rho_bc', 'delta_rho',
                  'beta_dqn', 'beta_bc', 'delta_beta',
                  'p_perm_dqn', 'p_perm_bc', 'p_perm_delta']
    print(df[print_cols].to_string(index=False))

    # ── 10. Run 평균 summary (lag별) ──────────────────────────────────────
    numeric_cols = [c for c in df.columns
                    if c not in ('subject', 'roi', 'file_idx', 'lag_tr')]
    df_summary = (df.groupby(['roi', 'lag_tr'])[numeric_cols]
                    .mean().reset_index())
    summary_path = out_dir / f'rsa_summary_{args.subject}.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f'Saved summary → {summary_path}')

    # ── 11. 시각화 ────────────────────────────────────────────────────────
    # lag=0 기준 요약 막대 그래프
    df_lag0 = df_summary[df_summary['lag_tr'] == 0].reset_index(drop=True)
    if not df_lag0.empty:
        save_summary_plot(df_lag0, out_dir, args.subject)
    else:
        print('  WARNING: lag=0 결과 없음. 막대 그래프 생략.')

    # lag sweep 라인 플롯
    save_lag_plot(df_summary, out_dir, args.subject)

    print(f'\n{"=" * 65}')
    print(f'Done. Outputs in: {out_dir}')


if __name__ == '__main__':
    main()
