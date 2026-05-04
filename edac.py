"""
edac.py — EDAC-style Ensemble Offline Actor-Critic for Atari (discrete actions)

This is NOT imitation learning.
  - Human actions are offline transition data used ONLY for Bellman backup.
  - No BC loss, no imitation constraint of any kind.
  - Diversity is driven by bootstrap masking + EDAC-style regularizer on critics.
  - The resulting epistemic uncertainty reflects Q-value disagreement across heads,
    NOT human behavioral uncertainty. Validate behaviorally before interpreting.

Architecture:
  - Actor  : π(a|s) = softmax(Linear(3136→512)→ReLU→Linear(512→A))
  - Critics: H independent Q-heads (same architecture as actor head)
  - CNN    : shared frozen backbone from pretrained DQN
  - Target : hard-copy of critics; no target actor needed (discrete SAC)

Training:
  - Critic loss : SAC-discrete Bellman
      V(s') = Σ_a π(a|s')[Q_target_mean(s',a') - α log π(a|s')]
      y_h   = r + γ(1-d)V(s')
      L_c_h = smooth_l1(Q_h(s, a_data), y_h)
  - Actor  loss : L_a = Σ_a π(a|s)[α log π(a|s) - Q_mean_or_min(s,a)]
  - Alpha  loss : L_α = -α(H(π) - target_entropy)  [if learn_alpha]
  - Div    loss : EDAC advantage_orthogonal or policy_js on critic ensemble
"""

import argparse
import copy
import glob
import random
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from glm_utils import load_dqn

_ROOT = Path(__file__).parent


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='EDAC ensemble offline actor-critic')
    # Data
    p.add_argument('--data_dir',    default=str(_ROOT / 'processed_data_frameskip_4'))
    p.add_argument('--subject',     default='sub_1')
    p.add_argument('--file_idx',    nargs='+', type=int,
                   default=[0,1,2,3,4,5,6,7,8,9,10])
    # Model paths
    p.add_argument('--dqn_path',    default=str(_ROOT / 'pretrained' / 'dqn_cnn.pt'))
    p.add_argument('--out_path',    default=str(_ROOT / 'trained_models_edac' / 'sub_1_edac_ens.pth'))
    p.add_argument('--freeze_cnn',  action='store_true', default=True)
    # Ensemble
    p.add_argument('--num_heads',   type=int,   default=10)
    # Training
    p.add_argument('--batch_size',  type=int,   default=256)
    p.add_argument('--epochs',      type=int,   default=20)
    p.add_argument('--lr',          type=float, default=1e-4,
                   help='Critic (and CNN if unfreeze) learning rate')
    p.add_argument('--actor_lr',    type=float, default=1e-4)
    p.add_argument('--gamma',       type=float, default=0.99)
    p.add_argument('--target_update_interval', type=int, default=1000)
    p.add_argument('--iters_per_epoch', type=int, default=0,
                   help='0 = auto (min_head_len // batch_size)')
    # Bootstrap
    p.add_argument('--bootstrap_frac', type=float, default=0.7)
    p.add_argument('--bootstrap_mode', choices=['frame','block','run'], default='block')
    p.add_argument('--block_size',  type=int,   default=512)
    # Entropy temperature
    p.add_argument('--alpha',       type=float, default=0.1,
                   help='Entropy temperature (fixed if --learn_alpha not set)')
    p.add_argument('--learn_alpha', action='store_true', default=False)
    p.add_argument('--alpha_lr',    type=float, default=3e-4)
    p.add_argument('--target_entropy', type=float, default=None,
                   help='Target policy entropy. Default: -ln(1/A) * 0.5')
    # Actor Q pessimism
    p.add_argument('--q_pessimism', choices=['mean','min'], default='min',
                   help='Q aggregation for actor update (min = conservative)')
    # Diversity
    p.add_argument('--div_type',    choices=['advantage_orthogonal','policy_js','none'],
                   default='advantage_orthogonal')
    p.add_argument('--lambda_div',  type=float, default=0.01)
    p.add_argument('--target_js',   type=float, default=0.03)
    p.add_argument('--div_tau',     type=float, default=1.0,
                   help='Softmax temperature for policy_js diversity')
    # Misc
    p.add_argument('--grad_clip',   type=float, default=10.0)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save_every',  type=int,   default=0)
    return p.parse_args()


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data loading ──────────────────────────────────────────────────────────────

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
            print(f'  WARNING: file_idx {fidx} out of range, skip')
            continue
        data   = np.load(npz_files[fidx])
        states = data['state']                          # (T, 4, 84, 84) uint8
        acts   = data['action']                         # (T, 6) one-hot or (T,) int
        rews   = data['reward'].astype(np.float32)
        T      = len(states)

        if acts.ndim == 2:
            acts = acts.argmax(axis=1)
        acts = acts.astype(np.int64)

        d        = np.zeros(T - 1, dtype=np.float32)
        d[-1]    = 1.0
        all_s.append(states[:-1]);  all_ns.append(states[1:])
        all_a.append(acts[:-1]);    all_r.append(rews[:-1])
        all_d.append(d)
        all_rid.append(np.full(T - 1, run_id, dtype=np.int32))
        print(f'  run {run_id} (f{fidx}): {T:,} frames → {T-1:,} transitions')

    ds = OfflineDataset(
        states      = np.concatenate(all_s),
        actions     = np.concatenate(all_a),
        rewards     = np.concatenate(all_r),
        next_states = np.concatenate(all_ns),
        dones       = np.concatenate(all_d),
        run_ids     = np.concatenate(all_rid),
    )
    print(f'  Total transitions: {ds.N:,}')
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
                # FIX: ceil so last partial block is not missed
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

class Actor(nn.Module):
    """
    Stochastic policy π(a|s) for discrete actions.
    Returns (probs, log_probs) both (B, A) using log_softmax for stability.
    """
    def __init__(self, feat_dim: int = 3136, hidden: int = 512, n_actions: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_actions)
        )

    def forward(self, feat: torch.Tensor):
        logits    = self.net(feat)                       # (B, A)
        log_probs = F.log_softmax(logits, dim=-1)        # (B, A)
        return log_probs.exp(), log_probs                # probs, log_probs


def build_critics(num_heads: int) -> nn.ModuleList:
    """Each critic head: Linear(3136→512)→ReLU→Linear(512→6)."""
    return nn.ModuleList([
        nn.Sequential(nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, 6))
        for _ in range(num_heads)
    ])


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(cnn, x: torch.Tensor) -> torch.Tensor:
    """(B, 4, 84, 84) float32 [0,1] → (B, 3136)"""
    h = F.relu(cnn.conv1(x))
    h = F.relu(cnn.conv2(h))
    h = F.relu(cnn.conv3(h))
    return h.view(h.size(0), -1)


def extract_features_grad(cnn, x: torch.Tensor) -> torch.Tensor:
    h = F.relu(cnn.conv1(x))
    h = F.relu(cnn.conv2(h))
    h = F.relu(cnn.conv3(h))
    return h.view(h.size(0), -1)


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
    s  = torch.from_numpy(dataset.states[idx].astype(np.float32) / 255.0).to(device)
    return s


# ── Critic loss (SAC-discrete Bellman) ────────────────────────────────────────

def compute_critic_loss(head_h, target_heads: nn.ModuleList, actor: Actor,
                         feat_s: torch.Tensor, feat_ns: torch.Tensor,
                         a: torch.Tensor, r: torch.Tensor, d: torch.Tensor,
                         gamma: float, alpha: float) -> torch.Tensor:
    """
    SAC-discrete target:
      V(s') = Σ_a π(a'|s') [Q_target_mean(s',a') - α log π(a'|s')]
      y     = r + γ(1-d) V(s')
    No BC loss. Bellman backup only.
    """
    q_sa = head_h(feat_s).gather(1, a.unsqueeze(1)).squeeze(1)    # (B,)

    with torch.no_grad():
        next_probs, next_log_probs = actor(feat_ns)                # (B, A)
        q_next_all = torch.stack(
            [th(feat_ns) for th in target_heads], dim=1
        )                                                           # (B, H, A)
        q_next_mean = q_next_all.mean(dim=1)                       # (B, A)
        # Soft value: weighted sum over actions
        v_next = (next_probs * (q_next_mean - alpha * next_log_probs)).sum(dim=-1)  # (B,)
        y      = r + gamma * (1.0 - d) * v_next                   # (B,)

    return F.smooth_l1_loss(q_sa, y)


# ── Actor loss ────────────────────────────────────────────────────────────────

def compute_actor_loss(actor: Actor, critics: nn.ModuleList,
                        feat_s: torch.Tensor,
                        alpha: float, q_pessimism: str):
    """
    L_actor = Σ_a π(a|s) [α log π(a|s) - Q(s,a)]
    Q is detached from critics (no gradient into critic on actor step).
    """
    probs, log_probs = actor(feat_s)                               # (B, A)
    with torch.no_grad():
        q_all = torch.stack([h(feat_s) for h in critics], dim=1)  # (B, H, A)
        q_for_actor = (q_all.min(dim=1).values if q_pessimism == 'min'
                       else q_all.mean(dim=1))                     # (B, A)
    loss           = (probs * (alpha * log_probs - q_for_actor)).sum(dim=-1).mean()
    policy_entropy = -(probs * log_probs).sum(dim=-1).mean().item()
    return loss, policy_entropy


# ── Alpha loss (learnable temperature) ───────────────────────────────────────

def compute_alpha_loss(log_alpha: torch.Tensor, probs: torch.Tensor,
                        log_probs: torch.Tensor, target_entropy: float):
    """L_α = -α · (H(π) - H_target)"""
    with torch.no_grad():
        entropy = -(probs * log_probs).sum(dim=-1)                 # (B,)
    return -(log_alpha * (entropy - target_entropy).detach()).mean()


# ── EDAC diversity regularizer ────────────────────────────────────────────────

def compute_diversity_loss(critics: nn.ModuleList, feat: torch.Tensor,
                            div_type: str, div_tau: float,
                            target_js: float, eps: float = 1e-8) -> torch.Tensor:
    """
    advantage_orthogonal:
      Penalizes cosine similarity between per-head advantage vectors.
      Encourages heads to disagree on which actions are relatively better.

    policy_js:
      Squared deviation of mean JS divergence from target_js.
      Prevents both collapse and unbounded divergence.
    """
    H = len(critics)

    if div_type == 'advantage_orthogonal':
        q_all = torch.stack([h(feat) for h in critics], dim=1)    # (B, H, A)
        adv   = q_all - q_all.mean(dim=-1, keepdim=True)           # (B, H, A)
        adv_n = adv / adv.norm(dim=-1, keepdim=True).clamp(min=eps)# (B, H, A)
        cos   = torch.einsum('bha,bka->bhk', adv_n, adv_n)         # (B, H, H)
        mask  = torch.triu(torch.ones(H, H, device=feat.device, dtype=torch.bool),
                           diagonal=1)
        return (cos[:, mask] ** 2).mean()

    elif div_type == 'policy_js':
        q_all   = torch.stack([h(feat) for h in critics], dim=1)  # (B, H, A)
        probs   = F.softmax(q_all / div_tau, dim=-1)               # (B, H, A)
        p_bar   = probs.mean(dim=1)                                # (B, A)
        h_total = -(p_bar * (p_bar + eps).log()).sum(dim=-1)       # (B,)
        h_heads = -(probs * (probs + eps).log()).sum(dim=-1)       # (B, H)
        js      = (h_total - h_heads.mean(dim=1)).mean()
        return (js - target_js) ** 2

    else:
        return torch.tensor(0.0, device=feat.device)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_uncertainty(critics: nn.ModuleList, actor: Actor,
                          cnn, dataset: OfflineDataset,
                          device: str, n_eval: int = 10_000):
    """
    Ensemble diagnostics:
      total/aleatoric/epistemic entropy, vote entropy, agreement,
      pairwise JS (stabilized), actor policy entropy.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(dataset.N, min(n_eval, dataset.N), replace=False)
    s   = torch.from_numpy(
        dataset.states[idx].astype(np.float32) / 255.0
    ).to(device)

    feat  = extract_features(cnn, s)                               # (B, 3136)
    q_all = torch.stack([h(feat) for h in critics], dim=1)        # (B, H, A)
    probs = F.softmax(q_all, dim=-1)                               # (B, H, A)
    p_bar = probs.mean(dim=1)                                      # (B, A)
    eps   = 1e-12
    H_num = len(critics)
    A     = q_all.size(-1)

    h_total = -(p_bar * (p_bar + eps).log()).sum(dim=-1)           # (B,)
    h_heads = -(probs  * (probs  + eps).log()).sum(dim=-1)         # (B, H)
    h_alea  = h_heads.mean(dim=1)                                  # (B,)
    h_epi   = h_total - h_alea                                     # (B,)

    # Vote entropy + agreement
    top_acts = q_all.argmax(dim=-1)                                # (B, H)
    vote_cnt = torch.zeros(len(s), A, device=device)
    vote_cnt.scatter_add_(1, top_acts,
                           torch.ones_like(top_acts, dtype=torch.float))
    vote_p   = vote_cnt / H_num
    vote_ent = -(vote_p * (vote_p + eps).log()).sum(dim=-1)        # (B,)
    agreement= vote_p.max(dim=-1).values                           # (B,)

    # Pairwise JS — numerically stable via log differences
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
    pairwise_js = torch.stack(pairs_js, dim=1).mean(dim=1)         # (B,)

    # Actor policy entropy
    actor_probs, actor_log = actor(feat)
    actor_ent = -(actor_probs * actor_log).sum(dim=-1)             # (B,)

    def _stats(t, name):
        t = t.cpu().float()
        q = torch.quantile(t, torch.tensor([.5, .9, .95]))
        print(f'  {name:<28s}: mean={t.mean():.4f}  std={t.std():.4f}  '
              f'p50={q[0]:.4f}  p90={q[1]:.4f}  p95={q[2]:.4f}')

    print(f'  [Eval {len(s):,} states | max H={np.log(A):.4f}]')
    _stats(h_total,     'total entropy')
    _stats(h_alea,      'aleatoric')
    _stats(h_epi,       'epistemic')
    _stats(vote_ent,    'vote entropy')
    _stats(agreement,   'agreement')
    _stats(pairwise_js, 'pairwise JS')
    _stats(actor_ent,   'actor policy entropy')


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = args.device
    rng    = np.random.default_rng(args.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f'\n[Data] {args.subject}  files={args.file_idx}')
    dataset = load_offline_transitions(args.data_dir, args.subject, args.file_idx)

    print(f'\n[Bootstrap] mode={args.bootstrap_mode}  frac={args.bootstrap_frac}')
    heads_idx = build_bootstrap_indices(
        dataset, args.num_heads, args.bootstrap_frac,
        args.bootstrap_mode, args.block_size, rng,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    A = 6  # Atari action space size
    print(f'\n[Model] CNN: {args.dqn_path}')
    cnn = load_dqn(args.dqn_path)
    cnn.to(device).eval()
    if args.freeze_cnn:
        for p_param in cnn.parameters():
            p_param.requires_grad_(False)

    critics        = build_critics(args.num_heads).to(device)
    target_critics = copy.deepcopy(critics).to(device)
    for p_param in target_critics.parameters():
        p_param.requires_grad_(False)

    actor = Actor(n_actions=A).to(device)

    # ── Optimizers ────────────────────────────────────────────────────────────
    critic_params = list(critics.parameters())
    if not args.freeze_cnn:
        critic_params += list(cnn.parameters())
    critic_opt = torch.optim.Adam(critic_params, lr=args.lr)
    actor_opt  = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # Alpha (entropy temperature)
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
    global_step     = 0

    print(f'\n[Train] epochs={args.epochs}  iters/epoch={iters_per_epoch}')
    print(f'        div={args.div_type}  λ_div={args.lambda_div}'
          f'  α={alpha:.3f}  learn_α={args.learn_alpha}'
          f'  q_pessimism={args.q_pessimism}')
    print(f'        device={device}  freeze_cnn={args.freeze_cnn}\n')

    for epoch in tqdm(range(1, args.epochs + 1)):
        critics.train(); actor.train()
        if not args.freeze_cnn:
            cnn.train()

        log = dict(td=[], div=[], actor=[], alpha=[], policy_ent=[], q_mean=[])

        for _ in range(iters_per_epoch):
            if args.learn_alpha:
                alpha = log_alpha.exp().item()

            # ── Critic update ──────────────────────────────────────────────
            td_sum      = torch.tensor(0.0, device=device)
            feat_s_list = []

            for h in range(args.num_heads):
                s, a, r, ns, d = sample_batch(dataset, heads_idx[h],
                                              args.batch_size, device)
                feat_fn = (extract_features if args.freeze_cnn
                           else extract_features_grad)
                feat_s  = feat_fn(cnn, s)
                feat_ns = feat_fn(cnn, ns)

                td = compute_critic_loss(
                    critics[h], target_critics, actor,
                    feat_s, feat_ns, a, r, d,
                    args.gamma, alpha,
                )
                td_sum = td_sum + td
                feat_s_list.append(feat_s.detach())
                log['q_mean'].append(critics[h](feat_s.detach()).detach().mean().item())

            td_mean = td_sum / args.num_heads

            # Diversity loss on concatenated features from all heads
            feat_div = torch.cat(feat_s_list, dim=0)               # (H*B, 3136)
            div_loss = compute_diversity_loss(
                critics, feat_div,
                args.div_type, args.div_tau, args.target_js,
            )

            critic_loss = td_mean + args.lambda_div * div_loss
            critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in critic_opt.param_groups[0]['params'] if p.requires_grad],
                args.grad_clip,
            )
            critic_opt.step()

            # ── Actor update ───────────────────────────────────────────────
            # Use a fresh global batch (avoids coupling with critic bootstrap)
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
                a_loss = compute_alpha_loss(log_alpha, probs_a, log_probs_a,
                                            target_entropy)
                alpha_opt.zero_grad()
                a_loss.backward()
                alpha_opt.step()
                alpha_loss_val = a_loss.item()
                alpha = log_alpha.exp().item()

            # ── Target update ──────────────────────────────────────────────
            global_step += 1
            if global_step % args.target_update_interval == 0:
                target_critics.load_state_dict(critics.state_dict())

            log['td'].append(td_mean.item())
            log['div'].append(div_loss.item())
            log['actor'].append(actor_loss.item())
            log['alpha'].append(alpha_loss_val)
            log['policy_ent'].append(policy_ent)

        # ── Epoch diagnostics ──────────────────────────────────────────────
        critics.eval(); actor.eval()
        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'td={np.mean(log["td"]):.4f}  '
              f'div={np.mean(log["div"]):.6f}  '
              f'actor={np.mean(log["actor"]):.4f}  '
              f'α={alpha:.4f}  '
              f'H_π={np.mean(log["policy_ent"]):.4f}  '
              f'Q_mean={np.mean(log["q_mean"]):.3f}')
        evaluate_uncertainty(critics, actor, cnn, dataset, device)

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt = Path(args.out_path).with_suffix(f'.epoch{epoch:03d}.pth')
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(critics.state_dict(), ckpt)
            print(f'  Checkpoint -> {ckpt}')

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(critics.state_dict(), out_path)
    print(f'\n[Done] Critics saved -> {out_path}')

    actor_path = out_path.with_stem(out_path.stem + '_actor')
    torch.save(actor.state_dict(), actor_path)
    print(f'       Actor  saved -> {actor_path}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = get_args()
    print('=' * 60)
    print('EDAC Ensemble Offline Actor-Critic')
    print(f'  Subject     : {args.subject}')
    print(f'  Heads       : {args.num_heads}')
    print(f'  Epochs      : {args.epochs}')
    print(f'  Diversity   : {args.div_type}  λ={args.lambda_div}')
    print(f'  Bootstrap   : {args.bootstrap_mode}  frac={args.bootstrap_frac}')
    print(f'  Alpha       : {args.alpha}  learn={args.learn_alpha}')
    print(f'  Q pessimism : {args.q_pessimism}')
    print(f'  Device      : {args.device}')
    print('=' * 60)
    train(args)


if __name__ == '__main__':
    main()
