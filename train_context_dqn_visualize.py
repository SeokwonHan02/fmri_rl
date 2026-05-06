#!/usr/bin/env python3
"""
train_context_dqn_visualize.py — Interactive frame-by-frame DQN episode viewer.

Records one episode using a loaded DQN policy (and optionally a MC risk detector
+ expert intervention), then opens a matplotlib window for step-by-step inspection.

Each frame shows:
  • Original ALE colour render (pre-action state)
  • Controller badge: LEARNER / EXPERT / COOLDOWN
  • Action name + index
  • Q-values for every action (horizontal bar chart, selected action highlighted)
  • Lives, per-step reward, cumulative episode return
  • MC risk bars — overall + left-/right-/noop-biased groups (if enabled)
  • Freshness indicator: ✦ when MC risk was just recomputed this step

Keyboard navigation
───────────────────
  →  /  Space  /  .   : next step
  ←  /  ,            : previous step
  Page Down           : +20 steps
  Page Up             : −20 steps
  Home                : first step
  End                 : last step
  q  /  Escape        : quit

Slider at the bottom can also be dragged to scrub.

──────────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────────
# Expert policy, MC risk overlay (delayed-life window):
python train_context_dqn_visualize.py \\
    --checkpoint pretrained/dqn_cnn.pt \\
    --mode offense \\
    --use_mc_risk_detector \\
    --mc_risk_group_size 4 \\
    --mc_risk_horizon_steps 13 \\
    --mc_risk_hit_life_delay_frames 120

# Trained learner checkpoint, greedy:
python train_context_dqn_visualize.py \\
    --checkpoint checkpoints/offense/offense_dqn_final.pt \\
    --mode offense \\
    --epsilon 0.0

# With expert intervention enabled:
python train_context_dqn_visualize.py \\
    --checkpoint checkpoints/offense/offense_dqn_final.pt \\
    --expert_checkpoint pretrained/dqn_cnn.pt \\
    --mode offense \\
    --use_expert_intervention \\
    --use_mc_risk_detector \\
    --mc_risk_threshold 0.4
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from PIL import Image

# ── matplotlib ────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

# Pillow resampling compat
try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]

# ── ALE registration ──────────────────────────────────────────────────────────
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass


# =============================================================================
# 1.  DQN  (layer names match pretrained/dqn_cnn.pt exactly)
# =============================================================================

class DQN(nn.Module):
    def __init__(self, n_actions: int = 6):
        super().__init__()
        self.conv1  = nn.Conv2d(4,  32, kernel_size=8, stride=4)
        self.conv2  = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3  = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc3    = nn.Linear(3136, 512)
        self.fc_out = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc3(x))
        return self.fc_out(x)


# =============================================================================
# 2.  Environment factory  (identical to eval.py / train_dqn.py)
# =============================================================================

def make_env(env_id: str = 'SpaceInvadersNoFrameskip-v4', seed: int = 0) -> gym.Env:
    env = gym.make(env_id, render_mode='rgb_array')
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    env.reset(seed=seed)
    return env


# =============================================================================
# 3.  Utilities
# =============================================================================

def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.array(obs, dtype=np.uint8)
    return torch.from_numpy(arr).unsqueeze(0).to(device).float() / 255.0


def get_lives(env: gym.Env) -> Optional[int]:
    try:
        return env.unwrapped.ale.lives()
    except Exception:
        return None


def load_model(path: str, n_actions: int, device: torch.device) -> DQN:
    net = DQN(n_actions=n_actions).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get('policy_net', ckpt)
    net.load_state_dict(state_dict)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def clone_ale_state(env: gym.Env):
    ale = env.unwrapped.ale
    for name in ('cloneState', 'clone_state'):
        if hasattr(ale, name):
            return getattr(ale, name)()
    raise RuntimeError("ALE does not support state cloning (upgrade ale-py).")


def restore_ale_state(env: gym.Env, state) -> None:
    ale = env.unwrapped.ale
    for name in ('restoreState', 'restore_state'):
        if hasattr(ale, name):
            getattr(ale, name)(state)
            return
    raise RuntimeError("ALE does not support state restoration (upgrade ale-py).")


# =============================================================================
# 4.  MC biased-rollout risk detector  (same logic as train_dqn.py)
# =============================================================================

class MCRiskDetector:
    def __init__(self, env_id: str, horizon_steps: int = 13, frame_skip: int = 4,
                 screen_size: int = 84, n_stack: int = 4, seed: int = 0,
                 group_size: int = 8, hit_life_delay_frames: int = 120):
        self.horizon_steps  = horizon_steps
        self.frame_skip     = frame_skip
        self.screen_size    = screen_size
        self.n_stack        = n_stack
        self.group_size     = group_size
        self.n_rollouts     = 3 * group_size
        self.risk_min_frame = hit_life_delay_frames
        self.risk_max_frame = hit_life_delay_frames + horizon_steps * frame_skip

        action_window = horizon_steps * frame_skip
        print(
            f"[MCRisk] Delayed-life window:\n"
            f"         hit_life_delay_frames={hit_life_delay_frames}  (= risk_min_frame)\n"
            f"         action_window_frames={action_window}  (= horizon_steps × frame_skip)\n"
            f"         risk_event_window=[{self.risk_min_frame}, {self.risk_max_frame}]\n"
            f"         phase-1 (biased actions): {horizon_steps} steps × {frame_skip} = {action_window} raw frames\n"
            f"         phase-2 (NOOP wait):      {hit_life_delay_frames} raw frames\n"
            f"         total rollout:            {self.risk_max_frame} raw frames"
        )

        self._env        = self._make_raw_env(env_id, seed)
        self._action_map = self._resolve_actions()

        noop  = self._action_map['NOOP']
        left  = self._action_map['LEFT']
        right = self._action_map['RIGHT']
        self._noop = noop
        self._acts = [noop, left, right]
        self._group_weights = [
            [0.25, 0.50, 0.25],   # group 0: left-biased
            [0.25, 0.25, 0.50],   # group 1: right-biased
            [0.50, 0.25, 0.25],   # group 2: noop-biased
        ]

    @staticmethod
    def _make_raw_env(env_id: str, seed: int) -> gym.Env:
        try:
            env = (gym.make(env_id) if 'NoFrameskip' in env_id
                   else gym.make(env_id, frameskip=1, repeat_action_probability=0.0))
        except TypeError:
            env = gym.make(env_id)
        env.reset(seed=seed + 99_999)
        return env

    def _resolve_actions(self) -> dict:
        meanings = self._env.unwrapped.get_action_meanings()
        m: dict = {'NOOP': None, 'LEFT': None, 'RIGHT': None}
        for i, name in enumerate(meanings):
            if name in m:
                m[name] = i
        missing = [k for k, v in m.items() if v is None]
        if missing:
            raise ValueError(f"MC risk: actions {missing} missing from {list(enumerate(meanings))}")
        return m

    def _get_frame(self) -> np.ndarray:
        ale = self._env.unwrapped.ale
        screen = ale.getScreenGrayscale()
        if screen.ndim == 3:
            screen = screen[:, :, 0]
        return np.array(
            Image.fromarray(screen).resize((self.screen_size, self.screen_size), _BILINEAR),
            dtype=np.uint8,
        )

    def _is_game_over(self, ale) -> bool:
        try:
            return bool(ale.game_over())
        except TypeError:
            return bool(ale.game_over(with_truncation=False))

    def _run_one_rollout(self, ale_state, start_lives: int, group_idx: int) -> int:
        self._env.reset()
        restore_ale_state(self._env, ale_state)
        ale = self._env.unwrapped.ale
        init_frame = self._get_frame()
        frame_stack = deque([init_frame.copy()] * self.n_stack, maxlen=self.n_stack)

        elapsed_raw_frames = 0

        # ── Phase 1: biased action probe ──────────────────────────────────────
        for _ in range(self.horizon_steps):
            action = random.choices(self._acts, weights=self._group_weights[group_idx], k=1)[0]
            for _ in range(self.frame_skip):
                self._env.step(action)
                elapsed_raw_frames += 1

                if ale.lives() < start_lives or self._is_game_over(ale):
                    if self.risk_min_frame <= elapsed_raw_frames <= self.risk_max_frame:
                        return 1
                    else:
                        return 0

            frame_stack.append(self._get_frame())

        # ── Phase 2: NOOP wait to observe delayed life losses ─────────────────
        while elapsed_raw_frames < self.risk_max_frame:
            self._env.step(self._noop)
            elapsed_raw_frames += 1

            if ale.lives() < start_lives or self._is_game_over(ale):
                if self.risk_min_frame <= elapsed_raw_frames <= self.risk_max_frame:
                    return 1
                else:
                    return 0

        return 0

    def estimate_risk_from_env(self, train_env: gym.Env) -> tuple:
        """Returns (overall, risk_left, risk_right, risk_noop). Does NOT step train_env."""
        ale_state   = clone_ale_state(train_env)
        start_lives = get_lives(train_env) or 3
        hits = [0, 0, 0]
        for g in range(3):
            for _ in range(self.group_size):
                hits[g] += self._run_one_rollout(ale_state, start_lives, g)

        risk_left  = hits[0] / self.group_size
        risk_right = hits[1] / self.group_size
        risk_noop  = hits[2] / self.group_size
        overall = max(risk_left, risk_right, risk_noop)

        return (overall, risk_left, risk_right, risk_noop)

    def close(self) -> None:
        self._env.close()


# =============================================================================
# 5.  FrameRecord — one decision step of recorded data
# =============================================================================

@dataclass
class FrameRecord:
    step: int              # decision-step index (0-based)
    env_frames: int        # raw ALE frames elapsed
    rgb: np.ndarray        # (H, W, 3) uint8 — pre-action colour frame
    action: int            # action index
    q_values: np.ndarray   # (n_actions,) float32
    reward: float          # immediate reward after this action
    ep_return: float       # cumulative reward so far (including this step)
    lives: int
    controller: str        # 'LEARNER' | 'EXPERT' | 'COOLDOWN'
    epsilon: float
    mc_risk: float
    mc_risk_left: float
    mc_risk_right: float
    mc_risk_noop: float
    mc_risk_fresh: bool    # True when MC was recomputed this step
    done: bool


# =============================================================================
# 6.  Episode recorder
# =============================================================================

def record_episode(args) -> tuple[List[FrameRecord], List[str]]:
    """
    Run one episode and collect a FrameRecord per decision step.

    The training env is never modified by MC risk evaluation — only its ALE
    state is cloned into a separate raw env for rollouts.
    """
    FRAME_SKIP = 4
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Environment ───────────────────────────────────────────────────────────
    env = make_env(args.env_id, seed=args.seed)
    n_actions = env.action_space.n
    action_meanings = env.unwrapped.get_action_meanings()
    print(f"[Env]    {args.env_id}  |  actions: {list(enumerate(action_meanings))}")

    # ── Policy ────────────────────────────────────────────────────────────────
    if args.checkpoint:
        policy = load_model(args.checkpoint, n_actions, device)
        print(f"[Policy] Loaded '{args.checkpoint}'  (ε={args.epsilon})")
    else:
        policy = DQN(n_actions=n_actions).to(device)
        policy.eval()
        print(f"[Policy] Random initialization (no checkpoint)  (ε={args.epsilon})")

    # ── Expert (optional) ─────────────────────────────────────────────────────
    expert: Optional[DQN] = None
    if args.use_expert_intervention and args.expert_checkpoint:
        if args.expert_checkpoint == args.checkpoint:
            expert = policy   # same weights; just shows what expert would do
        else:
            expert = load_model(args.expert_checkpoint, n_actions, device)
        print(f"[Expert] Loaded '{args.expert_checkpoint}'")

    # ── MC risk detector ──────────────────────────────────────────────────────
    mc_detector: Optional[MCRiskDetector] = None
    mc_eval_interval_steps = 0
    if args.use_mc_risk_detector:
        mc_detector = MCRiskDetector(
            env_id                = args.env_id,
            horizon_steps         = args.mc_risk_horizon_steps,
            frame_skip            = args.mc_risk_frame_skip,
            group_size            = args.mc_risk_group_size,
            seed                  = args.seed,
            hit_life_delay_frames = args.mc_risk_hit_life_delay_frames,
        )
        mc_eval_interval_steps = args.mc_risk_eval_interval_frames // FRAME_SKIP
        print(
            f"[MCRisk] N={3*args.mc_risk_group_size} rollouts × "
            f"H={args.mc_risk_horizon_steps} steps | "
            f"eval every {mc_eval_interval_steps} steps | "
            f"threshold={args.mc_risk_threshold}"
        )

    # ── Intervention step counts (mirroring train_dqn.py) ────────────────────
    offense_intervention_steps = args.offense_intervention_frames // FRAME_SKIP  # 10
    defense_intervention_steps = args.defense_intervention_frames // FRAME_SKIP  # 20
    defense_period_steps       = args.defense_period_frames       // FRAME_SKIP  # 100

    # ── Initial state ─────────────────────────────────────────────────────────
    obs, _ = env.reset(seed=args.seed)
    prev_lives = get_lives(env)

    env_frames     = 0
    decision_steps = 0
    ep_return      = 0.0

    expert_steps_remaining   = 0
    cooldown_steps_remaining = 0
    next_defense_takeover    = defense_period_steps

    cached_mc_risk       = 0.0
    cached_mc_risk_left  = 0.0
    cached_mc_risk_right = 0.0
    cached_mc_risk_noop  = 0.0
    next_mc_eval_step    = 0

    records: List[FrameRecord] = []
    max_steps = args.max_steps if args.max_steps else 50_000

    print(f"[Record] Starting (max {max_steps} steps) …")
    t0 = time.time()

    while True:
        # ── Capture colour frame BEFORE action ────────────────────────────────
        #   This matches the obs the agent observed when selecting the action.
        rgb = env.render()
        if rgb is None:
            rgb = np.zeros((210, 160, 3), dtype=np.uint8)

        # ── Q-values ─────────────────────────────────────────────────────────
        obs_t = obs_to_tensor(obs, device)
        with torch.no_grad():
            q_vals = policy(obs_t).squeeze(0).cpu().numpy()   # (n_actions,)

        # ── Controller / intervention state machine ───────────────────────────
        mc_risk_fresh = False

        if not args.use_expert_intervention or expert is None:
            # No intervention: purely show MC risk without triggering rescues.
            controller = 'LEARNER'
            if mc_detector is not None and decision_steps >= next_mc_eval_step:
                t_mc = time.time()
                (cached_mc_risk, cached_mc_risk_left,
                 cached_mc_risk_right, cached_mc_risk_noop) = \
                    mc_detector.estimate_risk_from_env(env)
                next_mc_eval_step = decision_steps + mc_eval_interval_steps
                mc_risk_fresh = True
                print(
                    f"  [MCRisk] step={decision_steps}  "
                    f"overall={cached_mc_risk:.3f}  "
                    f"L={cached_mc_risk_left:.2f}  "
                    f"R={cached_mc_risk_right:.2f}  "
                    f"N={cached_mc_risk_noop:.2f}  "
                    f"({time.time()-t_mc:.2f}s)"
                )

        elif args.mode == 'offense':
            if expert_steps_remaining > 0:
                # Intervention in progress.
                controller = 'EXPERT'
                expert_steps_remaining -= 1
                if expert_steps_remaining == 0:
                    cooldown_steps_remaining = args.post_intervention_cooldown_steps
            else:
                # Refresh MC risk.
                if mc_detector is not None and decision_steps >= next_mc_eval_step:
                    t_mc = time.time()
                    (cached_mc_risk, cached_mc_risk_left,
                     cached_mc_risk_right, cached_mc_risk_noop) = \
                        mc_detector.estimate_risk_from_env(env)
                    next_mc_eval_step = decision_steps + mc_eval_interval_steps
                    mc_risk_fresh = True
                    print(
                        f"  [MCRisk] step={decision_steps}  "
                        f"overall={cached_mc_risk:.3f}  "
                        f"({time.time()-t_mc:.2f}s)"
                    )

                # Rescue decision.
                do_rescue = (
                    (cached_mc_risk > args.mc_risk_threshold)
                    if args.use_mc_risk_detector else False
                )

                if do_rescue and random.random() < args.offense_intervention_prob:
                    controller = 'EXPERT'
                    expert_steps_remaining = offense_intervention_steps - 1
                elif cooldown_steps_remaining > 0:
                    controller = 'COOLDOWN'
                    cooldown_steps_remaining -= 1
                else:
                    controller = 'LEARNER'

        else:   # defense periodic intervention
            if expert_steps_remaining > 0:
                controller = 'EXPERT'
                expert_steps_remaining -= 1
                if expert_steps_remaining == 0:
                    cooldown_steps_remaining = args.post_intervention_cooldown_steps
                    next_defense_takeover    = decision_steps + defense_period_steps
            elif cooldown_steps_remaining > 0:
                controller = 'COOLDOWN'
                cooldown_steps_remaining -= 1
            elif decision_steps >= next_defense_takeover:
                controller = 'EXPERT'
                expert_steps_remaining = defense_intervention_steps - 1
            else:
                controller = 'LEARNER'

        # ── Select action ─────────────────────────────────────────────────────
        if controller == 'EXPERT' and expert is not None:
            with torch.no_grad():
                action = expert(obs_t).argmax(dim=1).item()
        else:
            if random.random() < args.epsilon:
                action = random.randrange(n_actions)
            else:
                action = int(q_vals.argmax())

        # ── Step ─────────────────────────────────────────────────────────────
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        current_lives = get_lives(env)
        if current_lives is None:
            current_lives = prev_lives or 3
        ep_return += float(reward)

        # ── Record ────────────────────────────────────────────────────────────
        records.append(FrameRecord(
            step          = decision_steps,
            env_frames    = env_frames,
            rgb           = rgb.copy(),
            action        = action,
            q_values      = q_vals.copy(),
            reward        = float(reward),
            ep_return     = ep_return,
            lives         = current_lives,
            controller    = controller,
            epsilon       = args.epsilon,
            mc_risk       = cached_mc_risk,
            mc_risk_left  = cached_mc_risk_left,
            mc_risk_right = cached_mc_risk_right,
            mc_risk_noop  = cached_mc_risk_noop,
            mc_risk_fresh = mc_risk_fresh,
            done          = done,
        ))

        prev_lives     = current_lives
        obs            = next_obs
        env_frames    += FRAME_SKIP
        decision_steps += 1

        if done or decision_steps >= max_steps:
            break

    env.close()
    if mc_detector is not None:
        mc_detector.close()

    elapsed = time.time() - t0
    print(
        f"[Record] {len(records)} steps  |  "
        f"return={ep_return:.1f}  |  {elapsed:.1f}s"
    )
    return records, action_meanings


# =============================================================================
# 7.  Interactive Visualizer
# =============================================================================

# Visual theme
_BG_DARK   = '#0F0F1A'
_BG_PANEL  = '#161626'
_BORDER    = '#2A2A44'
_TEXT_DIM  = '#888899'
_TEXT_MAIN = '#DDDDEE'

_CTRL_STYLE = {
    'LEARNER':  dict(bg='#1565C0', fg='white'),   # blue
    'EXPERT':   dict(bg='#B71C1C', fg='white'),   # red
    'COOLDOWN': dict(bg='#E65100', fg='white'),   # deep orange
}

_Q_SEL   = '#FF8C00'   # selected action bar
_Q_POS   = '#3A8FD4'   # positive Q bar
_Q_NEG   = '#C0392B'   # negative Q bar
_RISK_HI = '#E53935'   # risk above threshold
_RISK_LO = '#43A047'   # risk below threshold


class Visualizer:
    """
    matplotlib-based interactive viewer.

    Layout (15 × 9 inches)
    ───────────────────────
    ┌──────────────┬──────────────────┐
    │              │  INFO (text)     │  row 0 — tall
    │  GAME FRAME  ├──────────────────┤
    │  (color)     │  Q-VALUES (bars) │  row 1 — medium
    │              ├──────────────────┤
    │              │  MC RISK  (bars) │  row 2 — medium (if mc_enabled)
    ├──────────────┴──────────────────┤
    │         SLIDER  (scrubber)      │  row 3 — thin
    └─────────────────────────────────┘
    """

    def __init__(
        self,
        records: List[FrameRecord],
        action_meanings: List[str],
        mc_enabled: bool = False,
        mc_threshold: float = 0.5,
    ):
        self.records         = records
        self.action_meanings = action_meanings
        self.mc_enabled      = mc_enabled
        self.mc_threshold    = mc_threshold
        self.idx             = 0

        self._build_figure()
        self._init_artists()
        self._draw(0)

    # ── Figure construction ───────────────────────────────────────────────────

    def _build_figure(self) -> None:
        n_data_rows  = 3 if self.mc_enabled else 2
        h_ratios     = ([5, 3, 2] if self.mc_enabled else [5, 3])

        self.fig = plt.figure(figsize=(15, 9), facecolor=_BG_DARK)
        try:
            self.fig.canvas.manager.set_window_title('DQN Episode Viewer')
        except Exception:
            pass

        gs = gridspec.GridSpec(
            nrows = n_data_rows + 1,   # +1 for slider row
            ncols = 2,
            figure       = self.fig,
            width_ratios = [3, 2],
            height_ratios= h_ratios + [1],
            hspace = 0.45,
            wspace = 0.35,
            left   = 0.04, right = 0.97,
            top    = 0.93, bottom= 0.09,
        )

        # Game frame — left column, spans all data rows
        self.ax_game = self.fig.add_subplot(gs[:n_data_rows, 0])
        # Info text — top right
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        # Q-values — middle right
        self.ax_q    = self.fig.add_subplot(gs[1, 1])
        # MC risk — bottom right (only if mc_enabled)
        self.ax_risk = self.fig.add_subplot(gs[2, 1]) if self.mc_enabled else None

        # Style all axes
        for ax in filter(None, [self.ax_game, self.ax_info, self.ax_q, self.ax_risk]):
            ax.set_facecolor(_BG_PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(_BORDER)

        # Slider (separate axes outside gridspec for precise placement)
        slider_ax = self.fig.add_axes(
            [0.06, 0.025, 0.88, 0.022],
            facecolor='#1E1E30',
        )
        self.slider = Slider(
            ax       = slider_ax,
            label    = '',
            valmin   = 0,
            valmax   = max(len(self.records) - 1, 1),
            valinit  = 0,
            valstep  = 1,
            color    = '#3A6EA5',
        )
        self.slider.valtext.set_color(_TEXT_DIM)
        self.slider.on_changed(self._on_slider)

        # Key events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self.ax_game.axis('off')
        self.ax_info.axis('off')

    def _init_artists(self) -> None:
        """Create persistent artists that are updated in-place (no ax.clear())."""
        rec = self.records[0]

        # Game frame imshow
        self._img = self.ax_game.imshow(
            rec.rgb,
            aspect='auto',
            interpolation='nearest',
        )

        # Controller overlay on game title
        self._title_game = self.ax_game.set_title(
            '', loc='left', fontsize=11, fontweight='bold', pad=4,
        )

        # Info text objects (created once, updated via set_text)
        # We'll recreate info text on each draw since positions vary; that's OK.

        # Q-value bars (created on first draw)
        self._q_bars   = None
        self._q_labels = None

        # Risk bars (created on first draw)
        self._risk_bars = None

    # ── Per-frame drawing ─────────────────────────────────────────────────────

    def _draw(self, idx: int) -> None:
        idx = max(0, min(idx, len(self.records) - 1))
        self.idx = idx
        rec = self.records[idx]

        self._draw_game(rec)
        self._draw_info(rec)
        self._draw_q(rec)
        if self.ax_risk is not None:
            self._draw_risk(rec)

        # Sync slider without re-triggering callback
        self.slider.eventson = False
        self.slider.set_val(idx)
        self.slider.eventson = True

        n = len(self.records)
        self.fig.suptitle(
            f'Step  {rec.step} / {n - 1}     '
            f'({rec.env_frames:,} env frames)     '
            f'Return  {rec.ep_return:.1f}',
            color='white', fontsize=11, fontweight='bold',
            x=0.5, y=0.975,
        )
        self.fig.canvas.draw_idle()

    def _draw_game(self, rec: FrameRecord) -> None:
        self._img.set_data(rec.rgb)

        style = _CTRL_STYLE[rec.controller]
        self.ax_game.set_title(
            f"  {rec.controller}  ",
            loc='left', fontsize=11, fontweight='bold',
            color=style['fg'],
            backgroundcolor=style['bg'],
            pad=4,
        )

    def _draw_info(self, rec: FrameRecord) -> None:
        ax = self.ax_info
        ax.clear()
        ax.set_facecolor(_BG_PANEL)
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor(_BORDER)

        action_name = (
            self.action_meanings[rec.action]
            if rec.action < len(self.action_meanings)
            else str(rec.action)
        )

        n_lives = max(0, rec.lives) if rec.lives else 0
        life_str = ('♥ ' * n_lives + '♡ ' * max(0, 3 - n_lives)).strip()

        ctrl_style = _CTRL_STYLE[rec.controller]

        def txt(y: float, content: str, color: str = _TEXT_MAIN,
                size: int = 10, bold: bool = False, bg: str = None):
            kw = dict(
                transform = ax.transAxes,
                color     = color,
                fontsize  = size,
                verticalalignment = 'top',
                fontfamily = 'monospace',
                fontweight = 'bold' if bold else 'normal',
            )
            if bg:
                kw['backgroundcolor'] = bg
            ax.text(0.06, y, content, **kw)

        y = 0.96
        txt(y, f'Action:  {action_name} ({rec.action})', '#FF9800', size=12, bold=True); y -= 0.13
        txt(y, f'Lives:   {life_str}', '#EF5350', size=10); y -= 0.10
        txt(y, f'Reward:  {rec.reward:+.1f}',
            '#4CAF50' if rec.reward >= 0 else '#EF5350', size=10); y -= 0.10
        txt(y, f'Return:  {rec.ep_return:.1f}', '#81C784', size=10); y -= 0.10
        txt(y, f'ε  =  {rec.epsilon:.3f}', _TEXT_DIM, size=9); y -= 0.12

        if self.mc_enabled:
            risk_col = _RISK_HI if rec.mc_risk > self.mc_threshold else _RISK_LO
            fresh    = '  ✦ fresh' if rec.mc_risk_fresh else '  cached'
            txt(y, '────────────────', _BORDER, size=8); y -= 0.09
            txt(y, f'MC Risk: {rec.mc_risk:.3f}{fresh}', risk_col, size=11, bold=True); y -= 0.11
            txt(y, f'  L-bias  {rec.mc_risk_left:.3f}', '#FF7043', size=9); y -= 0.09
            txt(y, f'  R-bias  {rec.mc_risk_right:.3f}', '#29B6F6', size=9); y -= 0.09
            txt(y, f'  N-bias  {rec.mc_risk_noop:.3f}', '#AB47BC', size=9); y -= 0.09

        # Controller badge at bottom
        ax.text(
            0.5, 0.02,
            f'  {rec.controller}  ',
            transform=ax.transAxes,
            color=ctrl_style['fg'], fontsize=10, fontweight='bold',
            ha='center', va='bottom',
            backgroundcolor=ctrl_style['bg'],
        )

    def _draw_q(self, rec: FrameRecord) -> None:
        ax = self.ax_q
        ax.clear()
        ax.set_facecolor(_BG_PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(_BORDER)

        n  = len(rec.q_values)
        ys = np.arange(n)
        q  = rec.q_values
        labels = (
            [self.action_meanings[i] for i in range(n)]
            if len(self.action_meanings) == n else [str(i) for i in range(n)]
        )
        colors = [
            _Q_SEL if i == rec.action else
            (_Q_NEG if q[i] < 0 else _Q_POS)
            for i in range(n)
        ]

        ax.barh(ys, q, color=colors, height=0.55, edgecolor='none')
        ax.axvline(0, color=_BORDER, linewidth=1.0, zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, color=_TEXT_MAIN, fontsize=8.5, fontfamily='monospace')
        ax.tick_params(axis='x', colors=_TEXT_DIM, labelsize=7)
        ax.set_xlabel('Q-value', color=_TEXT_DIM, fontsize=8)
        ax.set_title('Q-values', color=_TEXT_DIM, fontsize=9, pad=3)
        ax.invert_yaxis()

        # Annotate selected Q
        qa = q[rec.action]
        offset = abs(q.max() - q.min()) * 0.04 + 0.02
        ax.text(
            qa + (offset if qa >= 0 else -offset),
            rec.action,
            f'{qa:.2f}',
            va='center',
            ha='left' if qa >= 0 else 'right',
            color=_Q_SEL, fontsize=8.5, fontweight='bold',
        )

    def _draw_risk(self, rec: FrameRecord) -> None:
        ax = self.ax_risk
        ax.clear()
        ax.set_facecolor(_BG_PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(_BORDER)

        labels = ['Overall', 'L-bias', 'R-bias', 'N-bias']
        values = [rec.mc_risk, rec.mc_risk_left, rec.mc_risk_right, rec.mc_risk_noop]
        bar_colors = [
            '#FF7043' if values[0] > self.mc_threshold else '#66BB6A',  # overall
            '#FF7043', '#29B6F6', '#AB47BC',
        ]
        # dim bars that are below threshold
        bar_colors = [
            c if v > self.mc_threshold else
            ('#C62828' if v > self.mc_threshold else '#388E3C')
            for c, v in zip(bar_colors, values)
        ]
        bar_colors = [
            (_RISK_HI if v > self.mc_threshold else _RISK_LO) for v in values
        ]
        bar_colors[1] = '#EF6C00' if values[1] > self.mc_threshold else '#43A047'
        bar_colors[2] = '#0288D1' if values[2] > self.mc_threshold else '#00897B'
        bar_colors[3] = '#7B1FA2' if values[3] > self.mc_threshold else '#558B2F'

        ys = np.arange(len(labels))
        ax.barh(ys, values, color=bar_colors, height=0.45, edgecolor='none')
        ax.axvline(
            self.mc_threshold,
            color='#FFEE58', linewidth=1.3, linestyle='--', zorder=5,
            label=f'thr={self.mc_threshold}',
        )
        ax.set_xlim(0, 1)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, color=_TEXT_MAIN, fontsize=8.5, fontfamily='monospace')
        ax.tick_params(axis='x', colors=_TEXT_DIM, labelsize=7)
        ax.set_title('MC Risk', color=_TEXT_DIM, fontsize=9, pad=3)
        ax.invert_yaxis()

        # Value labels inside bars
        for i, v in enumerate(values):
            ax.text(
                min(v + 0.02, 0.97), i, f'{v:.3f}',
                va='center', ha='left',
                color='white', fontsize=8, fontweight='bold',
            )

        fresh_txt = '✦ fresh' if rec.mc_risk_fresh else 'cached'
        fresh_col = '#FFEE58' if rec.mc_risk_fresh else _TEXT_DIM
        ax.text(
            0.98, 0.03, fresh_txt,
            transform=ax.transAxes,
            ha='right', va='bottom',
            color=fresh_col, fontsize=8,
        )

    # ── Key + slider callbacks ────────────────────────────────────────────────

    def _on_key(self, event) -> None:
        k = event.key
        n = len(self.records)
        if   k in ('right', '.', ' ', 'space'):  self._draw(self.idx + 1)
        elif k in ('left',  ','):                 self._draw(self.idx - 1)
        elif k == 'pagedown':                     self._draw(min(self.idx + 20, n - 1))
        elif k == 'pageup':                       self._draw(max(self.idx - 20, 0))
        elif k == 'home':                         self._draw(0)
        elif k == 'end':                          self._draw(n - 1)
        elif k in ('q', 'escape'):                plt.close(self.fig)

    def _on_slider(self, val) -> None:
        self._draw(int(val))

    # ── Entry ─────────────────────────────────────────────────────────────────

    def show(self) -> None:
        print(
            "\nControls:\n"
            "  → / Space / .   next step\n"
            "  ← / ,           previous step\n"
            "  PgDn / PgUp     ± 20 steps\n"
            "  Home / End      first / last\n"
            "  q / Escape      quit\n"
        )
        plt.show(block=True)


# =============================================================================
# 8.  Argument parser
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Interactive frame-by-frame DQN episode viewer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core
    p.add_argument('--checkpoint', default=None,
                   help='DQN checkpoint to run (policy_net key, or raw state_dict). '
                        'Omit to use a randomly initialized network.')
    p.add_argument('--expert_checkpoint', default='pretrained/dqn_cnn.pt',
                   help='Expert checkpoint used for intervention')
    p.add_argument('--mode',   default='offense', choices=['offense', 'defense'])
    p.add_argument('--env_id', default='SpaceInvadersNoFrameskip-v4')
    p.add_argument('--seed',   type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # Episode settings
    p.add_argument('--epsilon',   type=float, default=0.0,
                   help='Exploration ε during recording (0 = greedy)')
    p.add_argument('--max_steps', type=int, default=None,
                   help='Max decision steps to record (None = full episode)')

    # Intervention (only used when --use_expert_intervention)
    p.add_argument('--use_expert_intervention', action='store_true',
                   help='Enable expert rescue / periodic takeover during recording')
    p.add_argument('--offense_intervention_prob',   type=float, default=0.9)
    p.add_argument('--offense_intervention_frames', type=int,   default=40)
    p.add_argument('--defense_period_frames',       type=int,   default=400)
    p.add_argument('--defense_intervention_frames', type=int,   default=80)
    p.add_argument('--post_intervention_cooldown_steps', type=int, default=4)

    # MC risk
    p.add_argument('--use_mc_risk_detector', action='store_true',
                   help='Run MC biased-rollout risk detector and display results')
    p.add_argument('--mc_risk_horizon_steps',           type=int,   default=10,
                   help='Action probe horizon in decision steps '
                        '(action_window = horizon_steps × frame_skip; default 13 → 52 raw frames)')
    p.add_argument('--mc_risk_eval_interval_frames',    type=int,   default=80)
    p.add_argument('--mc_risk_threshold',               type=float, default=0.4)
    p.add_argument('--mc_risk_frame_skip',              type=int,   default=4)
    p.add_argument('--mc_risk_group_size',              type=int,   default=8)
    p.add_argument('--mc_risk_hit_life_delay_frames',   type=int,   default=120,
                   help='Raw frames between bullet hit and ALE life-count decrease '
                        '(~120 for Space Invaders); sets risk_min_frame and Phase-2 length')

    return p.parse_args()


# =============================================================================
# 9.  Entry point
# =============================================================================

if __name__ == '__main__':
    args   = parse_args()
    records, action_meanings = record_episode(args)

    if not records:
        print("No frames recorded — nothing to display.")
        raise SystemExit(1)

    viz = Visualizer(
        records         = records,
        action_meanings = action_meanings,
        mc_enabled      = args.use_mc_risk_detector,
        mc_threshold    = args.mc_risk_threshold,
    )
    viz.show()
