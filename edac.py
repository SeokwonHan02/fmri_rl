"""
edac.py — EDAC-style Ensemble Offline Actor-Critic for Atari (discrete actions)

This is NOT imitation learning.
  - Human actions are offline transition data used ONLY for Bellman backup.
  - No BC loss, no imitation constraint of any kind.
  - Diversity is driven by bootstrap masking + EDAC-style gradient regularizer.
  - The resulting epistemic uncertainty reflects Q-value disagreement across heads,
    NOT human behavioral uncertainty. Validate behaviorally before interpreting.

Architecture:
  - Actor  : π(a|s) = softmax(Linear(3136→512)→ReLU→Linear(512→A))
  - Critics: H independent action-conditioned heads Q(s,a) = MLP([feat ‖ a_soft] → 1)
  - CNN    : shared frozen backbone from pretrained DQN
  - Target : soft update τ·θ + (1−τ)·θ_target each step; no target actor needed

Training:
  - Critic : SAC-discrete with ensemble-minimum target
      Q_target(s',a') = min_j Q_target_j(s', a')
      V(s')  = Σ_a π(a'|s') [min_j Q_target_j(s',a') − α log π(a'|s')]
      y_h    = r + γ(1−d) V(s')
      L_c_h  = smooth_l1(Q_h(s, a_data), y_h)
  - Actor  : L_a = Σ_a π(a|s)[α log π(a|s) − Q_min/mean(s,a)]
  - Alpha  : L_α = −α(H(π) − H_target)  [if --learn-alpha]
  - Div    : edac_grad — cosine² between per-critic ∂Q_i/∂a_relaxed vectors
             (fallbacks: advantage_orthogonal, policy_js)
"""

import argparse
import copy
import glob
import random
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).parent


# ── Utilities ─────────────────────────────────────────────────────────────────

def safe_save(obj, path: Path):
    """Atomic save: write to .tmp then rename to prevent EOCD corruption."""
    path = Path(path)
    tmp  = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='EDAC ensemble offline actor-critic')

    # Data (aligned with main.py)
    p.add_argument('--data-dir',    default=str(_ROOT / 'processed_data_frameskip_4'),
                   help='Base directory containing processed subject data')
    p.add_argument('--subject',     default='sub_1',
                   choices=['sub_1','sub_2','sub_3','sub_4','sub_5','sub_6'])
    p.add_argument('--file-idx',    nargs='+', type=int,
                   default=[0,1,2,3,4,5,6,7,8,9,10])

    # Model paths (aligned with main.py)
    p.add_argument('--dqn-path',    default=str(_ROOT / 'pretrained' / 'dqn_cnn.pt'),
                   help='Path to pretrained DQN CNN checkpoint')
    p.add_argument('--save-dir',    default=str(_ROOT / 'checkpoints'),
                   help='Root save dir; models saved to save_dir/edac/<subject>/')
    p.add_argument('--freeze-cnn',  action='store_true', default=True)

    # Ensemble
    p.add_argument('--num-heads',   type=int,   default=10)

    # Training (aligned with main.py)
    p.add_argument('--batch-size',  type=int,   default=256)
    p.add_argument('--epochs',      type=int,   default=20)
    p.add_argument('--lr',          type=float, default=1e-4,
                   help='Critic learning rate')
    p.add_argument('--actor-lr',    type=float, default=1e-4)
    p.add_argument('--gamma',       type=float, default=0.99)
    p.add_argument('--tau',               type=float, default=0.005,
                   help='Soft target update coefficient (Polyak: θ_target ← τθ + (1−τ)θ_target)')
    p.add_argument('--iters-per-epoch', type=int, default=0,
                   help='0 = auto (min_head_len // batch_size)')

    # Bootstrap
    p.add_argument('--bootstrap-frac', type=float, default=0.7)
    p.add_argument('--bootstrap-mode', choices=['frame','block','run'], default='block')
    p.add_argument('--block-size',  type=int,   default=512)

    # Entropy temperature
    p.add_argument('--alpha',       type=float, default=0.1,
                   help='SAC entropy temperature (fixed unless --learn-alpha)')
    p.add_argument('--learn-alpha', action='store_true', default=False)
    p.add_argument('--alpha-lr',    type=float, default=3e-4)
    p.add_argument('--target-entropy', type=float, default=None,
                   help='Target entropy; default −ln(1/A)×0.5')

    # Actor Q pessimism
    p.add_argument('--q-pessimism', choices=['mean','min'], default='min',
                   help='Q aggregation for actor update')

    # Diversity
    p.add_argument('--div-type',    default='edac_grad',
                   choices=['edac_grad','advantage_orthogonal','policy_js','none'])
    p.add_argument('--lambda-div',  type=float, default=1.0)
    p.add_argument('--cql-alpha',   type=float, default=0.0,
                   help='CQL conservative penalty weight (0 = disabled). '
                        'L_CQL = logsumexp_a Q(s,a) - Q(s, a_data)')
    p.add_argument('--target-js',   type=float, default=0.03,
                   help='Target JS for policy_js diversity')
    p.add_argument('--div-tau',     type=float, default=1.0,
                   help='Softmax temperature for policy_js diversity')
    p.add_argument('--div-batch-size', type=int, default=512,
                   help='Number of states used for diversity loss')
    p.add_argument('--tau-unc',     type=float, default=1.0,
                   help='Softmax temperature for uncertainty metrics  p = softmax(Q / tau_unc)')

    # Misc (aligned with main.py)
    p.add_argument('--grad-clip',   type=float, default=10.0)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save-interval', type=int, default=1,
                   help='Checkpoint save interval (epochs)')

    # Evaluation (aligned with main.py)
    p.add_argument('--env-name',      default='SpaceInvadersNoFrameskip-v4')
    p.add_argument('--eval-episodes', type=int,  default=30)
    p.add_argument('--eval-interval', type=int,  default=1,
                   help='Game evaluation interval (epochs)')
    p.add_argument('--deterministic', action='store_true', default=True)
    p.add_argument('--stochastic',    dest='deterministic', action='store_false')

    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dqn(dqn_path: str):
    """Load DQN CNN backbone from checkpoint. Returns model in eval mode."""
    import sys
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


class OfflineDataset:
    """Offline transitions. No cross-run boundary transitions."""
    def __init__(self, states, actions, rewards, next_states, dones, run_ids):
        self.states      = states       # (N, 4, 84, 84) uint8
        self.actions     = actions      # (N,) int64
        self.rewards     = rewards      # (N,) float32
        self.next_states = next_states  # (N, 4, 84, 84) uint8
        self.dones       = dones        # (N,) float32
        self.run_ids     = run_ids      # (N,) int32
        self.N           = len(states)


def load_offline_transitions(data_dir: str, subject: str,
                              file_idx: list) -> OfflineDataset:
    npz_files = sorted(glob.glob(str(Path(data_dir) / subject / '*.npz')))
    if not npz_files:
        raise FileNotFoundError(f'No npz files in {data_dir}/{subject}')

    all_s, all_a, all_r, all_ns, all_d, all_rid = [], [], [], [], [], []

    for run_id, fidx in enumerate(file_idx):
        if fidx < 0 or fidx >= len(npz_files):
            print(f'  WARNING: file_idx {fidx} out of range, skip', flush=True)
            continue
        data   = np.load(npz_files[fidx])
        states = data['state']
        acts   = data['action']
        rews   = data['reward'].astype(np.float32)
        T      = len(states)

        if acts.ndim == 2:
            acts = acts.argmax(axis=1)
        acts = acts.astype(np.int64)

        d     = np.zeros(T - 1, dtype=np.float32)
        d[-1] = 1.0
        all_s.append(states[:-1]);  all_ns.append(states[1:])
        all_a.append(acts[:-1]);    all_r.append(rews[:-1])
        all_d.append(d)
        all_rid.append(np.full(T - 1, run_id, dtype=np.int32))
        print(f'  run {run_id} (f{fidx}): {T:,} frames → {T-1:,} transitions', flush=True)

    ds = OfflineDataset(
        states      = np.concatenate(all_s),
        actions     = np.concatenate(all_a),
        rewards     = np.concatenate(all_r),
        next_states = np.concatenate(all_ns),
        dones       = np.concatenate(all_d),
        run_ids     = np.concatenate(all_rid),
    )
    print(f'  Total transitions: {ds.N:,}', flush=True)
    return ds


# ── Bootstrap indices ─────────────────────────────────────────────────────────

def build_bootstrap_indices(dataset: OfflineDataset,
                             num_heads: int, frac: float,
                             mode: str, block_size: int,
                             rng: np.random.Generator) -> list:
    N, run_ids, heads_idx = dataset.N, dataset.run_ids, []

    for h in range(num_heads):
        if mode == 'frame':
            idx = np.where(rng.random(N) < frac)[0]

        elif mode == 'block':
            idx_list = []
            for rid in np.unique(run_ids):
                pos = np.where(run_ids == rid)[0]
                n_blocks = int(np.ceil(len(pos) / block_size))
                for b in range(n_blocks):
                    if rng.random() < frac:
                        idx_list.append(pos[b * block_size:(b + 1) * block_size])
            idx = np.concatenate(idx_list) if idx_list else np.array([], dtype=np.int64)

        else:  # run
            idx_list = [np.where(run_ids == rid)[0]
                        for rid in np.unique(run_ids) if rng.random() < frac]
            idx = np.concatenate(idx_list) if idx_list else np.array([], dtype=np.int64)

        if len(idx) < 64:
            idx = rng.choice(N, max(64, int(N * frac)), replace=False)
            print(f'  WARNING head {h}: too few samples, resampled {len(idx)}')

        heads_idx.append(idx.astype(np.int64))
        print(f'  head {h}: {len(idx):,} / {N:,} ({100*len(idx)/N:.1f}%)')

    return heads_idx


# ── Networks ──────────────────────────────────────────────────────────────────

class CriticHead(nn.Module):
    """
    Action-conditioned Q-function: Q(s, a_soft) → scalar per sample.
    Input: concatenated [feat (3136) ‖ a_soft (A)] → hidden → 1.
    Used with one-hot actions for Bellman backup and with relaxed actions
    for EDAC gradient diversity.
    """
    def __init__(self, feat_dim: int = 3136, n_actions: int = 6, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # feat: (B, feat_dim),  a: (B, A) — one-hot or soft
        return self.net(torch.cat([feat, a], dim=-1)).squeeze(-1)  # (B,)


class Actor(nn.Module):
    """Stochastic policy π(a|s) for discrete actions."""
    def __init__(self, feat_dim: int = 3136, hidden: int = 512, n_actions: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_actions)
        )

    def forward(self, feat: torch.Tensor):
        log_probs = F.log_softmax(self.net(feat), dim=-1)
        return log_probs.exp(), log_probs   # probs (B,A), log_probs (B,A)


class EDACAgent:
    """Evaluation wrapper (CNN encoder + Actor) compatible with eval.evaluate_agent."""
    def __init__(self, cnn, actor: Actor, n_actions: int, device):
        self.cnn      = cnn
        self.actor    = actor
        self.n_actions = n_actions
        self.device   = device

    def eval(self):
        self.cnn.eval()
        self.actor.eval()
        return self

    def get_action(self, state, deterministic: bool = True) -> int:
        # state: (4, 84, 84) float [0,1], already normalised by eval.py
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(np.asarray(state)).float()
        s = state.unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = extract_features(self.cnn, s)
            probs, _ = self.actor(feat)
        if deterministic:
            return probs.argmax(dim=-1).item()
        return torch.multinomial(probs, 1).item()


def build_critics(num_heads: int, n_actions: int = 6) -> nn.ModuleList:
    return nn.ModuleList([
        CriticHead(feat_dim=3136, n_actions=n_actions)
        for _ in range(num_heads)
    ])


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(cnn, x: torch.Tensor) -> torch.Tensor:
    """(B, 4, 84, 84) float [0,1] → (B, 3136)"""
    h = F.relu(cnn.conv1(x))
    h = F.relu(cnn.conv2(h))
    h = F.relu(cnn.conv3(h))
    return h.view(h.size(0), -1)


def extract_features_grad(cnn, x: torch.Tensor) -> torch.Tensor:
    h = F.relu(cnn.conv1(x))
    h = F.relu(cnn.conv2(h))
    h = F.relu(cnn.conv3(h))
    return h.view(h.size(0), -1)


# ── Action-conditioned Q helpers ──────────────────────────────────────────────

def q_values_all_actions(critic: CriticHead, feat: torch.Tensor) -> torch.Tensor:
    """
    Compute Q(s, a_i) for every discrete action a_i by passing one-hot vectors.
    Returns (B, A).
    """
    B, device = feat.size(0), feat.device
    A = 6
    eye = torch.eye(A, device=device)                     # (A, A)
    # Batch all actions at once: feat repeated A times
    feat_exp = feat.unsqueeze(1).expand(-1, A, -1).reshape(B * A, -1)  # (B*A, D)
    a_exp    = eye.unsqueeze(0).expand(B, -1, -1).reshape(B * A, A)    # (B*A, A)
    return critic(feat_exp, a_exp).reshape(B, A)           # (B, A)


# ── Batch sampling ────────────────────────────────────────────────────────────

def sample_batch(dataset: OfflineDataset, head_idx: np.ndarray,
                 batch_size: int, device: str):
    idx = np.random.choice(head_idx, batch_size, replace=len(head_idx) < batch_size)
    s  = torch.from_numpy(dataset.states[idx].astype(np.float32) / 255.0).to(device)
    a  = torch.from_numpy(dataset.actions[idx]).long().to(device)
    r  = torch.from_numpy(dataset.rewards[idx]).to(device)
    ns = torch.from_numpy(dataset.next_states[idx].astype(np.float32) / 255.0).to(device)
    d  = torch.from_numpy(dataset.dones[idx]).to(device)
    return s, a, r, ns, d


def sample_global_batch(dataset: OfflineDataset, batch_size: int, device: str):
    idx = np.random.choice(dataset.N, batch_size, replace=False)
    return torch.from_numpy(dataset.states[idx].astype(np.float32) / 255.0).to(device)


# ── Critic loss (SAC-discrete, ensemble-minimum target) ───────────────────────

def compute_critic_loss(head_h: CriticHead,
                         target_heads: nn.ModuleList,
                         actor: Actor,
                         feat_s: torch.Tensor, feat_ns: torch.Tensor,
                         a: torch.Tensor, r: torch.Tensor, d: torch.Tensor,
                         gamma: float, alpha: float,
                         n_actions: int = 6) -> torch.Tensor:
    """
    SAC-discrete Bellman backup with ensemble-minimum target Q:
      Q_target(s',a') = min_j Q_target_j(s', a')   [pessimistic target]
      V(s') = Σ_a π(a'|s') [Q_target_min(s',a') − α log π(a'|s')]
      y = r + γ(1−d) V(s')
    """
    a_oh = F.one_hot(a, num_classes=n_actions).float()    # (B, A)
    q_sa = head_h(feat_s, a_oh)                           # (B,)

    with torch.no_grad():
        next_probs, next_log = actor(feat_ns)              # (B, A)
        # (B, H, A) → min across heads → (B, A)
        q_next_all = torch.stack(
            [q_values_all_actions(th, feat_ns) for th in target_heads], dim=1
        )
        q_next_min = q_next_all.min(dim=1).values         # (B, A)
        v_next = (next_probs * (q_next_min - alpha * next_log)).sum(dim=-1)  # (B,)
        y = r + gamma * (1.0 - d) * v_next

    return F.smooth_l1_loss(q_sa, y)


# ── CQL conservative penalty ─────────────────────────────────────────────────

def compute_cql_loss(head_h: CriticHead, feat_s: torch.Tensor,
                      a: torch.Tensor, n_actions: int = 6) -> torch.Tensor:
    """
    CQL penalty for discrete actions (Kumar et al., 2020):
      L_CQL = E_s[ logsumexp_a Q(s,a) - Q(s, a_data) ]
    Pushes Q down on OOD actions and up on data actions, preventing
    the critic from overestimating out-of-distribution Q-values.
    Uses q_values_all_actions to evaluate all A discrete actions efficiently.
    """
    q_all = q_values_all_actions(head_h, feat_s)           # (B, A)
    logsumexp_q = torch.logsumexp(q_all, dim=-1)           # (B,)
    a_oh  = F.one_hot(a, num_classes=n_actions).float()
    q_sa  = head_h(feat_s, a_oh)                           # (B,)
    return (logsumexp_q - q_sa).mean()


# ── Actor loss ────────────────────────────────────────────────────────────────

def compute_actor_loss(actor: Actor, critics: nn.ModuleList,
                        feat_s: torch.Tensor,
                        alpha: float, q_pessimism: str):
    """L_a = Σ_a π(a|s)[α log π(a|s) − Q(s,a)], Q detached from critics."""
    probs, log_probs = actor(feat_s)                      # (B, A)
    with torch.no_grad():
        q_all = torch.stack(
            [q_values_all_actions(h, feat_s) for h in critics], dim=1
        )                                                  # (B, H, A)
        q_for_actor = (q_all.min(dim=1).values if q_pessimism == 'min'
                       else q_all.mean(dim=1))             # (B, A)
    loss          = (probs * (alpha * log_probs - q_for_actor)).sum(dim=-1).mean()
    policy_entropy = -(probs * log_probs).sum(dim=-1).mean().item()
    return loss, policy_entropy


# ── Alpha loss ────────────────────────────────────────────────────────────────

def compute_alpha_loss(log_alpha: torch.Tensor, probs: torch.Tensor,
                        log_probs: torch.Tensor, target_entropy: float):
    with torch.no_grad():
        entropy = -(probs * log_probs).sum(dim=-1)
    return -(log_alpha * (entropy - target_entropy).detach()).mean()


# ── EDAC action-gradient diversity ────────────────────────────────────────────

def compute_edac_action_gradient_diversity(
        critics: nn.ModuleList,
        feat: torch.Tensor,
        actor: Actor,
        n_actions: int,
        eps: float = 1e-8,
        action_source: str = 'actor') -> torch.Tensor:
    """
    EDAC-style diversity adapted for discrete actions (An et al., 2021).

    For discrete actions the policy-gradient analogue uses a relaxed action
    vector a_relaxed ∈ Δ^(A−1).  We treat a_relaxed as a leaf and differentiate
    Q_i(feat, a_relaxed) w.r.t. a_relaxed, giving a direction in action-simplex
    space.  Penalising cosine² between these directions across critic pairs
    encourages each critic to respond differently to the same action perturbation.

    a_relaxed = π(a|s).detach()  [action_source='actor', default]
             or uniform 1/A      [action_source='uniform']
    """
    H = len(critics)

    if action_source == 'actor':
        with torch.no_grad():
            probs, _ = actor(feat.detach())
        a_relaxed = probs.detach().clone()
    else:  # 'uniform'
        a_relaxed = torch.full((feat.size(0), n_actions), 1.0 / n_actions,
                                device=feat.device)

    a_relaxed = a_relaxed.requires_grad_(True)   # leaf tensor

    grads = []
    for critic in critics:
        q_i = critic(feat.detach(), a_relaxed).sum()
        grad_i = torch.autograd.grad(
            q_i, a_relaxed,
            create_graph=True,    # retain graph for critic backward pass
            retain_graph=True,
        )[0]                      # (B, A)
        grad_norm = grad_i / grad_i.norm(dim=-1, keepdim=True).clamp(min=eps)
        grads.append(grad_norm)

    loss  = torch.tensor(0.0, device=feat.device)
    count = 0
    for i in range(H):
        for j in range(i + 1, H):
            cos_ij = (grads[i] * grads[j]).sum(dim=-1)   # (B,)
            loss   = loss + (cos_ij ** 2).mean()
            count += 1

    return loss / max(count, 1)


# ── Diversity loss dispatcher ─────────────────────────────────────────────────

def compute_diversity_loss(critics: nn.ModuleList,
                            feat: torch.Tensor,
                            actor: Actor,
                            n_actions: int,
                            div_type: str,
                            div_tau: float,
                            target_js: float,
                            eps: float = 1e-8) -> torch.Tensor:
    if div_type == 'edac_grad':
        return compute_edac_action_gradient_diversity(
            critics, feat, actor, n_actions, eps=eps, action_source='actor'
        )

    elif div_type == 'advantage_orthogonal':
        q_all = torch.stack(
            [q_values_all_actions(h, feat) for h in critics], dim=1
        )                                                    # (B, H, A)
        adv   = q_all - q_all.mean(dim=-1, keepdim=True)
        adv_n = adv / adv.norm(dim=-1, keepdim=True).clamp(min=eps)
        cos   = torch.einsum('bha,bka->bhk', adv_n, adv_n)  # (B, H, H)
        H     = len(critics)
        mask  = torch.triu(torch.ones(H, H, device=feat.device, dtype=torch.bool),
                           diagonal=1)
        return (cos[:, mask] ** 2).mean()

    elif div_type == 'policy_js':
        q_all  = torch.stack(
            [q_values_all_actions(h, feat) for h in critics], dim=1
        )
        probs  = F.softmax(q_all / div_tau, dim=-1)
        p_bar  = probs.mean(dim=1)
        h_tot  = -(p_bar * (p_bar + eps).log()).sum(dim=-1)
        h_heads= -(probs  * (probs  + eps).log()).sum(dim=-1)
        js     = (h_tot - h_heads.mean(dim=1)).mean()
        return (js - target_js) ** 2

    else:  # 'none'
        return torch.tensor(0.0, device=feat.device)


# ── Uncertainty diagnostics ───────────────────────────────────────────────────

@torch.no_grad()

def evaluate_uncertainty(critics: nn.ModuleList, actor: Actor,
                          cnn, dataset: OfflineDataset,
                          device: str, n_eval: int = 10_000,
                          tau_unc: float = 1.0) -> float:
    """
    Print per-metric uncertainty statistics and return mean pairwise JS.
    Uses q_values_all_actions to build q_all (B, H, A) from action-conditioned critics.
    tau_unc : softmax temperature for all entropy/JS metrics (default 1.0 = no scaling).
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(dataset.N, min(n_eval, dataset.N), replace=False)
    s   = torch.from_numpy(dataset.states[idx].astype(np.float32) / 255.0).to(device)

    feat  = extract_features(cnn, s)
    q_all = torch.stack(
        [q_values_all_actions(h, feat) for h in critics], dim=1
    )
    
    q_cpu = q_all.detach().cpu().float()  # (B, H, A)

    # action-wise gap within each head
    top2 = q_cpu.topk(2, dim=-1).values
    q_gap = top2[:, :, 0] - top2[:, :, 1]

    # state-wise action range
    q_range = q_cpu.max(dim=-1).values - q_cpu.min(dim=-1).values

    # head disagreement per action
    q_head_std = q_cpu.std(dim=1).mean(dim=-1)  # (B,)

    print(
        f'  Q diagnostics: mean={q_cpu.mean():.4f} std={q_cpu.std():.4f} '
        f'min={q_cpu.min():.4f} max={q_cpu.max():.4f}',
        flush=True
    )
    print(
        f'  Q action gap: top1-top2 mean={q_gap.mean():.4f} '
        f'p90={torch.quantile(q_gap.flatten(), 0.90):.4f} '
        f'p95={torch.quantile(q_gap.flatten(), 0.95):.4f}',
        flush=True
    )
    print(
        f'  Q action range: mean={q_range.mean():.4f} '
        f'p90={torch.quantile(q_range.flatten(), 0.90):.4f} '
        f'p95={torch.quantile(q_range.flatten(), 0.95):.4f}',
        flush=True
    )
    print(
        f'  Q head disagreement: mean_head_std={q_head_std.mean():.4f} '
        f'p90={torch.quantile(q_head_std, 0.90):.4f}',
        flush=True
    )
                                                                  # (B, H, A)
    probs = F.softmax(q_all / tau_unc, dim=-1)                     # (B, H, A)
    p_bar = probs.mean(dim=1)                                      # (B, A)
    eps   = 1e-12
    H_num = len(critics)
    A     = q_all.size(-1)

    h_total = -(p_bar * (p_bar + eps).log()).sum(dim=-1)
    h_heads = -(probs  * (probs  + eps).log()).sum(dim=-1)
    h_alea  = h_heads.mean(dim=1)
    h_epi   = h_total - h_alea

    top_acts = q_all.argmax(dim=-1)
    vote_cnt = torch.zeros(len(s), A, device=device)
    vote_cnt.scatter_add_(1, top_acts,
                           torch.ones_like(top_acts, dtype=torch.float))
    vote_p    = vote_cnt / H_num
    vote_ent  = -(vote_p * (vote_p + eps).log()).sum(dim=-1)
    agreement = vote_p.max(dim=-1).values

    pairs_js = []
    for i in range(H_num):
        for j in range(i + 1, H_num):
            p_i = probs[:, i, :]
            p_j = probs[:, j, :]
            m   = 0.5 * (p_i + p_j)
            js  = 0.5 * (
                (p_i * ((p_i + eps).log() - (m + eps).log())).sum(-1) +
                (p_j * ((p_j + eps).log() - (m + eps).log())).sum(-1)
            )
            pairs_js.append(js)
    pairwise_js = torch.stack(pairs_js, dim=1).mean(dim=1)

    actor_probs, actor_log = actor(feat)
    actor_ent = -(actor_probs * actor_log).sum(dim=-1)

    def _stats(t, name):
        t = t.detach().cpu().float()
        q = torch.quantile(t, torch.tensor([.5, .9, .95]))
        print(
            f'  {name:<28s}: mean={t.mean():.4f}  std={t.std():.4f}  '
            f'p50={q[0]:.4f}  p90={q[1]:.4f}  p95={q[2]:.4f}',
            flush=True
        )

    print(f'  [Uncertainty | {len(s):,} states | max H={np.log(A):.4f}]', flush=True)
    _stats(h_total,     'total entropy')
    _stats(h_alea,      'aleatoric')
    _stats(h_epi,       'epistemic')
    _stats(vote_ent,    'vote entropy')
    _stats(agreement,   'agreement')
    _stats(pairwise_js, 'pairwise JS')
    _stats(actor_ent,   'actor policy entropy')
    return pairwise_js.mean().item()


# ── Action alignment diagnostics ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_action_alignment(cnn, actor: Actor,
                               dataset: OfflineDataset,
                               device: str, n_eval: int = 20_000):
    """
    Measures how well the actor's policy aligns with the human offline data.
    This is the primary indicator that the ensemble is operating near the
    human trajectory manifold — a prerequisite for meaningful uncertainty.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(dataset.N, min(n_eval, dataset.N), replace=False)

    s = torch.from_numpy(dataset.states[idx].astype(np.float32) / 255.0).to(device)
    a = torch.from_numpy(dataset.actions[idx]).long().to(device)

    feat            = extract_features(cnn, s)
    probs, log_probs = actor(feat)

    pred       = probs.argmax(dim=-1)
    acc        = (pred == a).float().mean().item()
    ce         = F.nll_loss(log_probs, a).item()
    human_prob = probs[torch.arange(len(a), device=device), a].mean().item()
    entropy    = -(probs * log_probs).sum(dim=-1).mean().item()

    print(f'  [Action alignment | {len(a):,} states]')
    print(f'    acc         = {acc:.4f}')
    print(f'    CE          = {ce:.4f}')
    print(f'    P(human a)  = {human_prob:.4f}')
    print(f'    actor H     = {entropy:.4f}')


# ── Game evaluation ───────────────────────────────────────────────────────────

def evaluate_game(cnn, actor: Actor, n_actions: int,
                   env_name: str, device, num_episodes: int,
                   seed: int, deterministic: bool) -> dict:
    """Run game episodes via eval.evaluate_agent and return reward stats."""
    from eval import evaluate_agent as _eval_agent
    agent = EDACAgent(cnn, actor, n_actions, device)
    agent.eval()
    return _eval_agent(agent, env_name, device, num_episodes, seed, deterministic)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = args.device
    rng    = np.random.default_rng(args.seed)
    A      = 6   # Atari discrete action space

    # ── Save directory (mirrors main.py: save_dir/algo/subject/) ──────────────
    save_dir = Path(args.save_dir) / 'edac' / args.subject
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'Models will be saved to: {save_dir}')

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f'\n[Data] {args.subject}  files={args.file_idx}')
    dataset = load_offline_transitions(args.data_dir, args.subject, args.file_idx)

    print(f'\n[Bootstrap] mode={args.bootstrap_mode}  frac={args.bootstrap_frac}')
    heads_idx = build_bootstrap_indices(
        dataset, args.num_heads, args.bootstrap_frac,
        args.bootstrap_mode, args.block_size, rng,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    print(f'\n[Model] CNN: {args.dqn_path}')
    cnn = load_dqn(args.dqn_path)
    cnn.to(device).eval()
    if args.freeze_cnn:
        for p in cnn.parameters():
            p.requires_grad_(False)

    critics        = build_critics(args.num_heads, n_actions=A).to(device)
    target_critics = copy.deepcopy(critics).to(device)
    for p in target_critics.parameters():
        p.requires_grad_(False)

    actor = Actor(n_actions=A).to(device)

    # ── Optimizers ────────────────────────────────────────────────────────────
    critic_params = list(critics.parameters())
    if not args.freeze_cnn:
        critic_params += list(cnn.parameters())
    critic_opt = torch.optim.Adam(critic_params, lr=args.lr)
    actor_opt  = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    alpha = args.alpha
    log_alpha = alpha_opt = None
    if args.learn_alpha:
        log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    target_entropy = (args.target_entropy if args.target_entropy is not None
                      else -np.log(1.0 / A) * 0.5)

    # ── Training loop ─────────────────────────────────────────────────────────
    min_head_len    = min(len(i) for i in heads_idx)
    iters_per_epoch = (args.iters_per_epoch if args.iters_per_epoch > 0
                       else max(1, min_head_len // args.batch_size))

    print(f'\n[Train] epochs={args.epochs}  iters/epoch={iters_per_epoch}')
    print(f'        div={args.div_type}  λ_div={args.lambda_div}'
          f'  α={alpha:.3f}  learn_α={args.learn_alpha}')
    print(f'        q_pessimism={args.q_pessimism}  τ={args.tau}'
          f'  device={device}  freeze_cnn={args.freeze_cnn}')
    print('=' * 80)

    for epoch in range(1, args.epochs + 1):
        print(f'\n{"="*80}', flush=True)
        print(f'[Epoch {epoch}/{args.epochs}] start', flush=True)
        
        critics.train(); actor.train()
        if not args.freeze_cnn:
            cnn.train()

        log = dict(td=[], cql=[], div=[], actor=[], alpha_loss=[], policy_ent=[], q_mean=[])

        for _ in range(iters_per_epoch):
            if args.learn_alpha:
                alpha = log_alpha.exp().item()

            # ── Critic update ──────────────────────────────────────────────
            td_sum  = torch.tensor(0.0, device=device)
            cql_sum = torch.tensor(0.0, device=device)
            feat_s_list = []

            for h in range(args.num_heads):
                s, a, r, ns, d = sample_batch(dataset, heads_idx[h],
                                               args.batch_size, device)
                feat_fn = extract_features if args.freeze_cnn else extract_features_grad
                feat_s  = feat_fn(cnn, s)
                feat_ns = feat_fn(cnn, ns)

                td = compute_critic_loss(
                    critics[h], target_critics, actor,
                    feat_s, feat_ns, a, r, d,
                    args.gamma, alpha, n_actions=A,
                )
                td_sum = td_sum + td

                if args.cql_alpha > 0.0:
                    cql_sum = cql_sum + compute_cql_loss(
                        critics[h], feat_s, a, n_actions=A
                    )

                feat_s_list.append(feat_s.detach())

                with torch.no_grad():
                    q_mean_val = q_values_all_actions(
                        critics[h], feat_s.detach()
                    ).mean().item()
                log['q_mean'].append(q_mean_val)

            td_mean   = td_sum  / args.num_heads
            cql_mean  = cql_sum / args.num_heads
            feat_div  = torch.cat(feat_s_list, dim=0)  # (H*B, 3136)

            if feat_div.size(0) > args.div_batch_size:
                div_idx  = torch.randperm(feat_div.size(0), device=feat_div.device)[:args.div_batch_size]
                feat_div = feat_div[div_idx]

            div_loss = compute_diversity_loss(
                critics, feat_div, actor, A,
                args.div_type, args.div_tau, args.target_js,
            )

            critic_loss = (td_mean
                           + args.lambda_div * div_loss
                           + args.cql_alpha  * cql_mean)
            critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                [p for g in critic_opt.param_groups for p in g['params']
                 if p.requires_grad],
                args.grad_clip,
            )
            critic_opt.step()

            # ── Actor update ───────────────────────────────────────────────
            s_global   = sample_global_batch(dataset, args.batch_size, device)
            feat_actor = (extract_features(cnn, s_global) if args.freeze_cnn
                          else extract_features_grad(cnn, s_global))

            actor_loss, policy_ent = compute_actor_loss(
                actor, critics, feat_actor, alpha, args.q_pessimism
            )
            actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
            actor_opt.step()

            # ── Alpha update ───────────────────────────────────────────────
            alpha_loss_val = 0.0
            if args.learn_alpha:
                with torch.no_grad():
                    probs_a, log_probs_a = actor(feat_actor)
                al = compute_alpha_loss(log_alpha, probs_a, log_probs_a, target_entropy)
                alpha_opt.zero_grad()
                al.backward()
                alpha_opt.step()
                alpha_loss_val = al.item()
                alpha = log_alpha.exp().item()

            # ── Soft target update (Polyak) ────────────────────────────────
            with torch.no_grad():
                for p_tgt, p_src in zip(target_critics.parameters(),
                                         critics.parameters()):
                    p_tgt.data.mul_(1.0 - args.tau).add_(args.tau * p_src.data)

            log['td'].append(td_mean.item())
            log['cql'].append(cql_mean.item())
            log['div'].append(div_loss.item())
            log['actor'].append(actor_loss.item())
            log['alpha_loss'].append(alpha_loss_val)
            log['policy_ent'].append(policy_ent)

        # ── Epoch summary ──────────────────────────────────────────────────
        critics.eval(); actor.eval()
        print(
            f'Epoch {epoch:3d}/{args.epochs}  '
            f'td={np.mean(log["td"]):.4f}  '
            f'cql={np.mean(log["cql"]):.4f}  '
            f'div={np.mean(log["div"]):.6f}  '
            f'actor={np.mean(log["actor"]):.4f}  '
            f'α={alpha:.4f}  '
            f'H_π={np.mean(log["policy_ent"]):.4f}  '
            f'Q_mean={np.mean(log["q_mean"]):.3f}',
            flush=True
        )

        evaluate_uncertainty(critics, actor, cnn, dataset, device, tau_unc=args.tau_unc)
        if args.div_type == 'edac_grad':
            print(f'  edac_grad_diversity (last iter) = {log["div"][-1]:.6f}', flush = True)
        evaluate_action_alignment(cnn, actor, dataset, device)

        # ── Game evaluation (aligned with main.py eval_interval) ──────────
        if epoch % args.eval_interval == 0:
            print(f"\n{'='*80}", flush = True)
            print(f'EVALUATION at Epoch {epoch}', flush = True)
            print(f"{'='*80}", flush = True)
            eval_stats = evaluate_game(
                cnn, actor, A, args.env_name, device,
                args.eval_episodes, args.seed + epoch, args.deterministic,
            )
            print(
                f"  Eval - Mean Reward: {eval_stats['mean_reward']:.2f}"
                f" ± {eval_stats['std_reward']:.2f}\n"
                f"         Min: {eval_stats['min_reward']:.1f},"
                f" Max: {eval_stats['max_reward']:.1f}\n"
                f"         Mean Length: {eval_stats['mean_length']:.1f}",
                flush = True
            )
            print(f"{'='*80}\n", flush = True)

        # ── Checkpoint (aligned with main.py save_interval) ───────────────
        if epoch % args.save_interval == 0:
            safe_save(critics.state_dict(), save_dir / f'epoch_{epoch:03d}_critics.pth')
            safe_save(actor.state_dict(),   save_dir / f'epoch_{epoch:03d}_actor.pth')
            print(f'  Saved checkpoint -> {save_dir}/epoch_{epoch:03d}_*.pth', flush = True)

    # ── Final save ────────────────────────────────────────────────────────────
    safe_save(critics.state_dict(), save_dir / 'critics_final.pth')
    safe_save(actor.state_dict(),   save_dir / 'actor_final.pth')
    print(f'\n[Done] Critics -> {save_dir}/critics_final.pth')
    print(f'       Actor   -> {save_dir}/actor_final.pth')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = get_args()
    print('=' * 80)
    print('EDAC Ensemble Offline Actor-Critic')
    print(f'  Subject     : {args.subject}')
    print(f'  Data dir    : {args.data_dir}')
    print(f'  Save dir    : {args.save_dir}/edac/{args.subject}/')
    print(f'  Heads       : {args.num_heads}')
    print(f'  Epochs      : {args.epochs}')
    print(f'  Diversity   : {args.div_type}  λ={args.lambda_div}')
    print(f'  Bootstrap   : {args.bootstrap_mode}  frac={args.bootstrap_frac}')
    print(f'  Alpha       : {args.alpha}  learn={args.learn_alpha}')
    print(f'  Q pessimism : {args.q_pessimism}')
    print(f'  Env         : {args.env_name}  episodes={args.eval_episodes}')
    print(f'  Device      : {args.device}')
    print('=' * 80)
    train(args)


if __name__ == '__main__':
    main()
