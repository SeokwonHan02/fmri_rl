"""
compute_glm_regressors.py

GLM regressor matrix 생성.

출력 CSV columns (순서):
  time                 game time [s]
  [intercept]          optional
  action               0-5 integer (human action index)
  action_change        1 if action changed from previous frame, else 0
  reward               raw reward value
  death                1 if done (death/life event), else 0
  frame_diff_mse       pixel-wise MSE between consecutive frames (last channel)
  u_pixel_ae           AE reconstruction MSE
  u_pixel_vae          VAE reconstruction MSE
  u_pixel_rnd          RND prediction error
  u_q_unc_kendall      1 - Kendall's W (rank disagreement)
  u_q_vote_disagree    1 - max_vote / n_heads
  u_q_zscore_std       Z-score std of ensemble top-1 action
  u_q_adv_gap_var      variance of (Q_top1 - Q_top2) across heads

사용 예:
  python compute_glm_regressors.py \\
      --subject sub_1 --file_idx 10 \\
      --ae_path  variance/pixel/ae_831_100.pth \\
      --vae_path variance/pixel/vae_830_100.pth \\
      --rnd_path variance/pixel/rnd_829_100.pth \\
      --ensemble_path variance/q-level/ensemble_825_70.pth \\
      --output regressors_sub1_f10.csv
"""

import sys
import argparse
import glob
import io
import pickle
import zipfile
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'variance' / 'pixel'))
sys.path.insert(0, str(_ROOT / 'variance' / 'q-level'))


# ─── Argument parsing ─────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='GLM regressor matrix 생성')

    # Data
    p.add_argument('--data_dir', default=str(_ROOT / 'processed_data_frameskip_4'))
    p.add_argument('--subject',  default='sub_1',
                   choices=['sub_1','sub_2','sub_3','sub_4','sub_5','sub_6'])
    p.add_argument('--file_idx', type=int, default=10,
                   help='npz 파일 인덱스 (0-based)')

    # Model paths
    p.add_argument('--dqn_path',      default=str(_ROOT / 'pretrained' / 'dqn_cnn.pt'),
                   help='Pretrained DQN CNN checkpoint (CNN encoder용)')
    p.add_argument('--ae_path',       default=str(_ROOT / 'variance/pixel/ae_831_100.pth'))
    p.add_argument('--vae_path',      default=str(_ROOT / 'variance/pixel/vae_830_100.pth'))
    p.add_argument('--rnd_path',      default=str(_ROOT / 'variance/pixel/rnd_829_100.pth'))
    p.add_argument('--ensemble_path', default=str(_ROOT / 'variance/q-level/ensemble_825_70.pth'))

    # Options
    p.add_argument('--intercept',   action='store_true', default=False,
                   help='intercept column(상수 1) 포함')
    p.add_argument('--batch_size',  type=int, default=256)
    p.add_argument('--device',      default='auto')
    p.add_argument('--output',      default=None,
                   help='출력 CSV 경로 (기본: regressors_{subject}_f{file_idx}.csv)')

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
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m  = AE(latent_dim=ck['latent_dim']).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    print(f'  AE  loaded: {Path(path).name}  '
          f'(epoch={ck["epoch"]}, latent_dim={ck["latent_dim"]})')
    return m


def load_vae(path: str, device: torch.device):
    from vae import VAE
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m  = VAE(latent_dim=ck['latent_dim']).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    print(f'  VAE loaded: {Path(path).name}  '
          f'(epoch={ck["epoch"]}, latent_dim={ck["latent_dim"]}, beta={ck["beta"]})')
    return m


def load_rnd(path: str, device: torch.device):
    from rnd import RND
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m  = RND(feature_dim=ck['feature_dim']).to(device)
    m.predictor.load_state_dict(ck['predictor'])
    m.target.load_state_dict(ck['target'])
    m.eval()
    print(f'  RND loaded: {Path(path).name}  '
          f'(epoch={ck["epoch"]}, feature_dim={ck["feature_dim"]})')
    return m


# ─── Ensemble DQN loader (zip EOCD 복구 포함) ────────────────────────────────

def _repair_zip_eocd(path: str) -> bytes:
    with open(path, 'rb') as f:
        data = f.read()
    total  = len(data)
    LFH_SZ = 30
    DD_SZ   = 16

    lfh_pos = sorted(i for i in range(total - 4) if data[i:i+4] == b'PK\x03\x04')
    entries = []
    for pos in lfh_pos:
        (_, ver, flags, comp, mt, md, _, _, _, fnl, exl) = \
            struct.unpack_from('<4sHHHHHIIIHH', data, pos)
        fn  = data[pos + LFH_SZ : pos + LFH_SZ + fnl].decode('utf-8', errors='replace')
        ds  = pos + LFH_SZ + fnl + exl
        entries.append((pos, fn, ver, flags, comp, mt, md, fnl, exl, ds))

    file_entries = []
    for i, (pos, fn, ver, flags, comp, mt, md, fnl, exl, ds) in enumerate(entries):
        nxt = entries[i+1][0] if i+1 < len(entries) else total
        cand = data[nxt - DD_SZ : nxt]
        if cand[:4] == b'PK\x07\x08':
            _, crc, sz, _ = struct.unpack('<4sIII', cand)
        else:
            import zlib
            sz  = nxt - ds
            crc = zlib.crc32(data[ds:ds+sz]) & 0xFFFFFFFF
        file_entries.append((pos, fn, ver, flags & ~0x08, comp, mt, md,
                              crc, fnl, exl, ds, sz))

    out = bytearray(data)
    for (pos, fn, ver, flags, comp, mt, md, crc, fnl, exl, ds, sz) in file_entries:
        struct.pack_into('<H', out, pos + 6,  flags)
        struct.pack_into('<I', out, pos + 14, crc)
        struct.pack_into('<I', out, pos + 18, sz)
        struct.pack_into('<I', out, pos + 22, sz)

    cd_off = total
    cd = bytearray()
    for (pos, fn, ver, flags, comp, mt, md, crc, fnl, exl, ds, sz) in file_entries:
        fb = fn.encode('utf-8')
        cd += struct.pack('<4sHHHHHHIIIHHHHHII',
            b'PK\x01\x02', 20, ver, flags, comp, mt, md,
            crc, sz, sz, len(fb), 0, 0, 0, 0, 0, pos) + fb

    n = len(file_entries)
    eocd = struct.pack('<4sHHHHIIH', b'PK\x05\x06', 0, 0, n, n, len(cd), cd_off, 0)
    return bytes(out) + bytes(cd) + bytes(eocd)


def _load_heads_sd(path: str) -> dict:
    try:
        sd = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict):
            return sd
    except Exception:
        pass

    buf = io.BytesIO(_repair_zip_eocd(path))
    with zipfile.ZipFile(buf, 'r') as zf:
        names  = zf.namelist()
        prefix = names[0].split('/')[0] + '/'
        pkl_d  = zf.read(f'{prefix}data.pkl')
        storages = {}
        for name in names:
            if name.startswith(f'{prefix}data/') and name != f'{prefix}data/':
                key = name.split('/')[-1]
                raw = zf.read(name)
                storages[key] = torch.from_numpy(
                    np.frombuffer(raw, dtype='<f4').copy())

    class _UP(pickle.Unpickler):
        def find_class(self, mod, nm):
            if mod == 'torch._utils' and nm == '_rebuild_tensor_v2':
                def rb(st, off, sz, str_, *a):
                    return st.as_strided(sz, str_, off).clone()
                return rb
            return super().find_class(mod, nm)
        def persistent_load(self, pid):
            _, _, key, _, n = pid
            if key not in storages:
                return torch.zeros(n)
            s = storages[key]
            if len(s) < n:
                s = torch.cat([s, torch.zeros(n - len(s))])
            return s

    return _UP(io.BytesIO(pkl_d)).load()


def load_ensemble(path: str, dqn_path: str, device: torch.device):
    from model.dqn import load_pretrained_cnn

    sd  = _load_heads_sd(path)
    # 완전한 head 수 감지
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
            return torch.stack([h(f) for h in self.heads], dim=1)  # (B,H,6)

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
        yield i, arr[i:i+batch_size]


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


def compute_q_uncertainty(model, states_uint8: np.ndarray,
                           batch_size: int, device: torch.device) -> dict:
    """
    states_uint8: (N, 4, 84, 84) uint8
    returns: dict of (N,) arrays for each of 4 uncertainty metrics
    """
    kw, vd, zs, ag = [], [], [], []
    with torch.no_grad():
        for _, batch in tqdm(list(batched(states_uint8, batch_size)),
                             desc='    q-uncertainty', leave=False):
            x = torch.from_numpy(batch).to(device).float() / 255.0
            q_all = model(x)          # (B, H, A)
            B, H, A = q_all.shape

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
            q_mu   = q_all.mean(dim=-1, keepdim=True)
            q_sig  = q_all.std(dim=-1, keepdim=True).clamp_min(1e-8)
            z_all  = (q_all - q_mu) / q_sig
            t1_idx = top1_a.view(B, 1, 1).expand(B, H, 1)
            zs.append(z_all.gather(-1, t1_idx).squeeze(-1).std(dim=-1).cpu().numpy())

            # 4. advantage gap variance (top1 - top2)
            _, top2_idx = mean_q.topk(2, dim=-1)
            q_t1 = q_all.gather(-1, top2_idx[:,0].view(B,1,1).expand(B,H,1)).squeeze(-1)
            q_t2 = q_all.gather(-1, top2_idx[:,1].view(B,1,1).expand(B,H,1)).squeeze(-1)
            ag.append((q_t1 - q_t2).var(dim=-1).cpu().numpy())

    return {
        'u_q_unc_kendall':   np.concatenate(kw),
        'u_q_vote_disagree': np.concatenate(vd),
        'u_q_zscore_std':    np.concatenate(zs),
        'u_q_adv_gap_var':   np.concatenate(ag),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = resolve_device(args.device)
    print(f'Device: {device}')

    # ── 1. Load npz ──────────────────────────────────────────────────────────
    npz_files = sorted(glob.glob(
        str(Path(args.data_dir) / args.subject / '*.npz')))
    if not npz_files:
        raise FileNotFoundError(f'No npz files in {args.data_dir}/{args.subject}')
    npz_path = npz_files[args.file_idx]
    print(f'\nLoading: {Path(npz_path).name}')

    data   = np.load(npz_path)
    time_  = data['time'].astype(np.float32)          # (N,)
    states = data['state']                             # (N, 4, 84, 84) uint8
    acts   = data['action']                            # (N, 6) one-hot
    rews   = data['reward'].astype(np.float32)         # (N,)
    dones  = data['done'].astype(np.float32)           # (N,)
    N      = len(time_)
    print(f'  Frames: {N:,}  |  time: {time_[0]:.2f}s – {time_[-1]:.2f}s')

    # ── 2. Basic regressors (no model needed) ────────────────────────────────
    action_idx    = acts.argmax(axis=1).astype(np.int32)             # (N,)
    action_change = np.concatenate([[0], (action_idx[1:] != action_idx[:-1]).astype(np.float32)])
    # one-hot action columns
    ACTION_NAMES  = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHT+FIRE', 'LEFT+FIRE']
    action_onehot = (action_idx[:, None] == np.arange(6)[None, :]).astype(np.float32)  # (N, 6)

    # frame-to-frame pixel MSE (last channel of state stack)
    frames_last = states[:, -1, :, :].astype(np.float32) / 255.0    # (N, 84, 84)
    frame_diff  = np.zeros(N, dtype=np.float32)
    diff        = frames_last[1:] - frames_last[:-1]
    frame_diff[1:] = (diff ** 2).mean(axis=(1, 2))

    # ── 3. Load pixel novelty models ─────────────────────────────────────────
    print('\nLoading pixel-novelty models...')
    ae_model  = load_ae(args.ae_path,   device)
    vae_model = load_vae(args.vae_path, device)
    rnd_model = load_rnd(args.rnd_path, device)

    print('\nComputing pixel novelty...')
    print('  AE:')
    u_ae  = compute_pixel_novelty(ae_model,  states, args.batch_size, device)
    print('  VAE:')
    u_vae = compute_pixel_novelty(vae_model, states, args.batch_size, device)
    print('  RND:')
    u_rnd = compute_pixel_novelty(rnd_model, states, args.batch_size, device)

    # free GPU memory
    del ae_model, vae_model, rnd_model
    if device.type in ('cuda', 'mps'):
        torch.mps.empty_cache() if device.type == 'mps' else torch.cuda.empty_cache()

    # ── 4. Load ensemble DQN ─────────────────────────────────────────────────
    print('\nLoading ensemble DQN...')
    ens_model = load_ensemble(args.ensemble_path, args.dqn_path, device)

    print('Computing Q-level uncertainty...')
    q_metrics = compute_q_uncertainty(ens_model, states, args.batch_size, device)

    # ── 5. Assemble DataFrame ────────────────────────────────────────────────
    print('\nAssembling regressor DataFrame...')
    cols = {'time': time_}

    if args.intercept:
        cols['intercept'] = np.ones(N, dtype=np.float32)

    for i, name in enumerate(ACTION_NAMES):
        cols[f'action_{name.lower().replace("+", "_")}'] = action_onehot[:, i]
    cols['action_change'] = action_change
    cols['reward']        = rews
    cols['terminal']      = dones
    cols['frame_diff_mse']= frame_diff
    cols['u_pixel_ae']    = u_ae
    cols['u_pixel_vae']   = u_vae
    cols['u_pixel_rnd']   = u_rnd
    cols.update(q_metrics)

    df = pd.DataFrame(cols)

    # ── 6. Save ──────────────────────────────────────────────────────────────
    if args.output is None:
        out_path = Path(f'regressors_{args.subject}_f{args.file_idx}.csv')
    else:
        out_path = Path(args.output)

    df.to_csv(out_path, index=False, float_format='%.6f')
    print(f'\nSaved → {out_path}  ({len(df)} rows × {len(df.columns)} cols)')
    print(f'Columns: {list(df.columns)}')
    print('\nSummary statistics:')
    print(df.describe().to_string())


if __name__ == '__main__':
    main()
