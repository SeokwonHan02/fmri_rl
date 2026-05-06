#!/usr/bin/env python3
"""
train_context_dqn.py — Offense / Defense DQN training for Atari Space Invaders.

Architecture and environment preprocessing are kept identical to
pretrained/dqn_cnn.pt and eval.py so that checkpoints are interchangeable.

──────────────────────────────────────────────────────────────────────────────
Quick start
──────────────────────────────────────────────────────────────────────────────
# Offense DQN with MC biased-rollout risk detector (delayed-life window):
python train_context_dqn.py \
    --mode offense \
    --expert_checkpoint pretrained/dqn_cnn.pt \
    --save_dir checkpoints/offense_mc_biased_delay \
    --use_mc_risk_detector \
    --mc_risk_group_size 8 \
    --mc_risk_horizon_steps 13 \
    --mc_risk_eval_interval_frames 80 \
    --mc_risk_threshold 0.5 \
    --mc_risk_hit_life_delay_frames 120

# Offense DQN with random rescue fallback (no MC detector):
python train_dqn.py \
    --mode offense \
    --expert_checkpoint pretrained/dqn_cnn.pt \
    --save_dir checkpoints/offense

# Defense DQN (survival-only; expert takes periodic demonstrations):
python train_dqn.py \
    --mode defense \
    --expert_checkpoint pretrained/dqn_cnn.pt \
    --save_dir checkpoints/defense

──────────────────────────────────────────────────────────────────────────────
Frame / step relationship
──────────────────────────────────────────────────────────────────────────────
AtariPreprocessing applies frame_skip=4 internally.
Each env.step() call advances 4 raw environment frames.

    env_frames = decision_steps × 4

     40 env frames  =  10 decision steps  (offense intervention window)
     80 env frames  =  20 decision steps  (defense intervention window)
    400 env frames  = 100 decision steps  (defense takeover period)
     80 env frames  =  20 decision steps  (MC risk eval interval, default)
"""

import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from PIL import Image

# Pillow resampling compat (Pillow < 10 uses Image.BILINEAR, ≥ 10 uses Image.Resampling.BILINEAR)
try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]

# ── ALE registration ──────────────────────────────────────────────────────────
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Warning: ale_py not found. Install with: pip install ale-py")
except Exception as e:
    print(f"Warning: Could not register ALE environments: {e}")


class DQN(nn.Module):
    """
    Nature DQN architecture.

    Input : (B, 4, 84, 84) float32 in [0, 1]
    Output: (B, n_actions)  Q-values
    """

    def __init__(self, n_actions: int = 6):
        super().__init__()
        self.conv1  = nn.Conv2d(4,  32, kernel_size=8, stride=4)
        self.conv2  = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3  = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc3    = nn.Linear(3136, 512)   # 64 * 7 * 7 = 3136
        self.fc_out = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc3(x))
        return self.fc_out(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: float) -> None:
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, acts, rews, nobs, dones = zip(*batch)
        return (
            np.stack(obs),                          # (B, 4, 84, 84) uint8
            np.array(acts,  dtype=np.int64),         # (B,)
            np.array(rews,  dtype=np.float32),       # (B,)
            np.stack(nobs),                          # (B, 4, 84, 84) uint8
            np.array(dones, dtype=np.float32),       # (B,)
        )

    def __len__(self) -> int:
        return len(self.buf)


def make_env(env_id: str = 'SpaceInvadersNoFrameskip-v4', seed: int = 0) -> gym.Env:
    env = gym.make(env_id, render_mode='rgb_array')
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,                # one env.step() = 4 raw ALE frames
        screen_size=84,
        terminal_on_life_loss=False, # life-loss handled manually
        grayscale_obs=True,
        scale_obs=False,             # keep uint8; divide by 255 when batching
    )
    env = FrameStackObservation(env, stack_size=4)
    env.reset(seed=seed)
    return env


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """(4,84,84) uint8 → (1,4,84,84) float32 in [0,1]."""
    arr = np.array(obs, dtype=np.uint8)   # copy in case obs is a LazyFrame
    return torch.from_numpy(arr).unsqueeze(0).to(device).float() / 255.0


def get_lives(env: gym.Env):
    """Return current ALE life count, or None if unavailable."""
    try:
        return env.unwrapped.ale.lives()
    except Exception:
        return None


def load_expert(path: str, n_actions: int, device: torch.device) -> DQN:
    """Load a pretrained DQN as a frozen greedy expert (ε = 0)."""
    net = DQN(n_actions=n_actions).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt['policy_net'] if 'policy_net' in ckpt else ckpt
    net.load_state_dict(state_dict)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    print(f"[Expert] Loaded '{path}'  (frozen, greedy ε=0)")
    return net


def epsilon_schedule(env_frames: int, args) -> float:
    ratio = min(env_frames / args.epsilon_decay_frames, 1.0)
    return args.epsilon_start + ratio * (args.epsilon_final - args.epsilon_start)


def greedy_action(net: DQN, obs_t: torch.Tensor) -> int:
    with torch.no_grad():
        return net(obs_t).argmax(dim=1).item()


def epsilon_greedy_action(net: DQN, obs_t: torch.Tensor,
                          epsilon: float, n_actions: int) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return greedy_action(net, obs_t)


def clone_ale_state(env: gym.Env):
    """Clone emulator state from env.unwrapped.ale. Returns an ALEState object."""
    ale = env.unwrapped.ale
    for name in ('cloneState', 'clone_state'):
        if hasattr(ale, name):
            return getattr(ale, name)()
    raise RuntimeError(
        "[MCRisk] ALE does not expose a state-clone method. "
        "Upgrade ale-py: pip install --upgrade ale-py"
    )


def restore_ale_state(env: gym.Env, state) -> None:
    """Restore a previously cloned ALEState into env.unwrapped.ale."""
    ale = env.unwrapped.ale
    for name in ('restoreState', 'restore_state'):
        if hasattr(ale, name):
            getattr(ale, name)(state)
            return
    raise RuntimeError(
        "[MCRisk] ALE does not expose a state-restore method. "
        "Upgrade ale-py: pip install --upgrade ale-py"
    )


# =============================================================================
#     Monte Carlo biased-rollout risk detector
#
#     Estimates death risk from the current emulator state without modifying
#     the training environment.
#
#     Algorithm
#     ─────────
#     1. Clone ALE state from the training env.
#     2. Run N = 3 × group_size rollouts in a separate raw env.
#        Each rollout restores the cloned state and samples actions from one of
#        three biased policies:
#
#          group 0 (left-biased):   P(LEFT)=0.50, P(RIGHT)=0.25, P(NOOP)=0.25
#          group 1 (right-biased):  P(LEFT)=0.25, P(RIGHT)=0.50, P(NOOP)=0.25
#          group 2 (noop-biased):   P(LEFT)=0.25, P(RIGHT)=0.25, P(NOOP)=0.50
#
#        Only LEFT, RIGHT, NOOP are used; FIRE and combined actions are excluded.
#     3. A rollout is a "risk event" if lives decrease or game_over() fires
#        within H decision steps.
#     4. risk = mean(risk_events) over all N rollouts.
#
#     Group-wise risks (risk_left, risk_right, risk_noop) are also returned for
#     future use as behavioural risk regressors in fMRI analyses.
#
#     The raw risk env uses frameskip=1 with manual frame-skip in the rollout
#     loop, bypassing AtariPreprocessing / FrameStackObservation so that
#     ALE clone/restore behaviour is explicit and reliable.
#
#     A manual (84×84 grayscale, 4-frame) frame stack is maintained per rollout
#     even though the current biased policy does not consume it.  This makes it
#     straightforward to swap in a neural-network rollout policy later.
#
#     EXTENDING TO A NEURAL-NETWORK ROLLOUT POLICY
#     ──────────────────────────────────────────────
#     In _run_one_rollout, the stacked observation is assembled each decision
#     step as:
#         stacked_obs = np.stack(list(frame_stack))  # (4, 84, 84) uint8
#     Replace the biased random action with:
#         obs_t  = obs_to_tensor(stacked_obs, device)
#         action = greedy_action(policy_net, obs_t)
# =============================================================================

class MCRiskDetector:
    """
    Monte Carlo biased-rollout death-risk estimator for offense DQN training.

    Risk definition (delayed-life window)
    ──────────────────────────────────────
    In Space Invaders, when the player is hit by a bullet the ALE life counter
    decreases ~120 raw frames later.  A rollout that detects a life loss before
    that delay has elapsed was almost certainly hit *before* the rollout started
    (pending death), not by the rollout's own actions.

    The rollout runs in two phases:
      Phase 1 (action probe): horizon_steps decision steps with biased random
        actions — covers action_window = horizon_steps × frame_skip raw frames.
      Phase 2 (NOOP wait): NOOP for the remaining hit_life_delay_frames raw
        frames, letting any pending hits from Phase 1 materialize.

    A risk event is defined as:

        life count decreases within [risk_min_frame, risk_max_frame]

    where:
        risk_min_frame = hit_life_delay_frames
        risk_max_frame = hit_life_delay_frames + horizon_steps × frame_skip

    Any life loss before risk_min_frame is treated as a pre-existing hit → 0.

    Parameters
    ----------
    env_id                : Gymnasium / ALE environment ID.
    horizon_steps         : Action probe horizon in decision steps (default 13).
                            Determines the causal window:
                            action_window_frames = horizon_steps × frame_skip.
    frame_skip            : Raw frames per decision step (default 4).
    screen_size           : Resize target for manual preprocessing (default 84).
    n_stack               : Frame stack depth (default 4).
    num_workers           : Reserved for future parallel rollouts; ignored.
    seed                  : RNG seed for the raw risk env.
    group_size            : Rollouts per policy group; total N = 3 × group_size.
    hit_life_delay_frames : Raw frames between a hit and the resulting life-loss
                            in ALE (default 120 for Space Invaders).
                            Equals risk_min_frame and the length of Phase 2.
    """

    def __init__(
        self,
        env_id: str,
        horizon_steps: int = 13,
        frame_skip: int = 4,
        screen_size: int = 84,
        n_stack: int = 4,
        num_workers: int = 0,
        seed: int = 0,
        group_size: int = 8,
        hit_life_delay_frames: int = 120,
    ):
        self.horizon_steps   = horizon_steps
        self.frame_skip      = frame_skip
        self.screen_size     = screen_size
        self.n_stack         = n_stack
        self.group_size      = group_size
        self.n_rollouts      = 3 * group_size
        self.risk_min_frame  = hit_life_delay_frames
        self.risk_max_frame  = hit_life_delay_frames + horizon_steps * frame_skip

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

        if num_workers > 1:
            print(
                "[MCRisk] Warning: parallel rollout workers are not implemented. "
                "Falling back to sequential evaluation."
            )

        self._env        = self._make_raw_env(env_id, seed)
        self._action_map = self._resolve_actions()

        noop = self._action_map['NOOP']
        left = self._action_map['LEFT']
        right = self._action_map['RIGHT']

        self._noop = noop
        self._acts = [noop, left, right]

        self._group_weights = [
            [0.25, 0.50, 0.25],
            [0.25, 0.25, 0.50],
            [0.50, 0.25, 0.25],
        ]

    @staticmethod
    def _make_raw_env(env_id: str, seed: int) -> gym.Env:
        """
        Raw env with frameskip=1 and no preprocessing wrappers.

        The risk rollout manages its own frame-skip loop so that we can check
        for life-loss after every single raw frame.

        Using a seed offset (+ 99_999) keeps the raw env's noop-reset sequence
        independent of the training env's seed.
        """
        try:
            if 'NoFrameskip' in env_id:
                # NoFrameskip envs already have frameskip=1 baked in.
                env = gym.make(env_id)
            else:
                env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0)
        except TypeError:
            # Fallback if the env does not accept frameskip kwarg.
            env = gym.make(env_id)
        env.reset(seed=seed + 99_999)
        return env

    def _resolve_actions(self) -> dict:
        """
        Map action names NOOP / LEFT / RIGHT to their gym action indices.

        Uses env.unwrapped.get_action_meanings() which returns a list such as:
            ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        If any of the required actions are missing, raises ValueError with the
        full action list so the user can diagnose env / action-set mismatches.
        """
        try:
            meanings = self._env.unwrapped.get_action_meanings()
        except AttributeError as exc:
            raise RuntimeError(
                f"[MCRisk] Cannot get action meanings from "
                f"{type(self._env.unwrapped).__name__}: {exc}"
            ) from exc

        print(f"[MCRisk] Action meanings: {list(enumerate(meanings))}")

        action_map: dict = {'NOOP': None, 'LEFT': None, 'RIGHT': None}
        for idx, name in enumerate(meanings):
            if name in action_map:
                action_map[name] = idx

        missing = [k for k, v in action_map.items() if v is None]
        if missing:
            raise ValueError(
                f"[MCRisk] Required actions {missing} not found in "
                f"{list(enumerate(meanings))}. "
                "Check --env_id and the ALE action set."
            )

        print(
            f"[MCRisk] Mapped: "
            f"NOOP={action_map['NOOP']}, "
            f"LEFT={action_map['LEFT']}, "
            f"RIGHT={action_map['RIGHT']}"
        )
        return action_map

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _get_frame(self) -> np.ndarray:
        """
        Read current ALE screen and return (screen_size, screen_size) uint8
        grayscale.  Uses ale.getScreenGrayscale() to avoid a colour conversion.
        """
        ale    = self._env.unwrapped.ale
        screen = ale.getScreenGrayscale()   # (H, W, 1) uint8
        if screen.ndim == 3:
            screen = screen[:, :, 0]        # (H, W) uint8
        img = Image.fromarray(screen).resize(
            (self.screen_size, self.screen_size), _BILINEAR
        )
        return np.array(img, dtype=np.uint8)

    # ── Rollout ────────────────────────────────────────────────────────────────

    def _is_game_over(self, ale) -> bool:
        """Robust game_over check across ale-py API versions."""
        try:
            return bool(ale.game_over())
        except TypeError:
            return bool(ale.game_over(with_truncation=False))

    def _run_one_rollout(self, ale_state, start_lives: int, group_idx: int) -> int:
        """
        Restore state and run one biased rollout in two phases.

        Phase 1 — action probe (horizon_steps decision steps)
            Samples biased random actions to explore the causal risk window.
            Covers the first action_window_frames = horizon_steps × frame_skip
            raw frames.

        Phase 2 — NOOP wait (hit_life_delay_frames raw frames)
            Holds NOOP to let any hits sustained during Phase 1 materialize as
            life-loss events.  No new actions are sampled in this phase.

        A risk event (return 1) is triggered when the life count drops or
        game-over occurs inside [risk_min_frame, risk_max_frame].
        Any life loss before risk_min_frame is treated as a pre-existing hit
        carried into the rollout from before its start → return 0.
        """
        # Reset wrapper state, then restore ALE checkpoint.
        self._env.reset()
        restore_ale_state(self._env, ale_state)

        ale     = self._env.unwrapped.ale
        weights = self._group_weights[group_idx]

        # Initialise frame stack (ready for future network-policy upgrades).
        init_frame = self._get_frame()
        frame_stack = deque(
            [init_frame.copy() for _ in range(self.n_stack)],
            maxlen=self.n_stack,
        )

        elapsed_raw_frames = 0

        # ── Phase 1: biased action probe ──────────────────────────────────────
        for _ in range(self.horizon_steps):
            action = random.choices(self._acts, weights=weights, k=1)[0]

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

        return 0   # no life loss in risk window

    # ── Public interface ───────────────────────────────────────────────────────

    def estimate_risk_from_env(self, train_env: gym.Env) -> tuple:
        """
        Clone ALE state from train_env and run 3 × group_size biased rollouts.

        train_env is NEVER stepped — only its ALE state is cloned at the start.
        After this call the training env remains exactly as it was.

        Returns
        ───────
        (overall_risk, risk_left, risk_right, risk_noop)
            All values are in [0, 1].
            risk_left  = fraction of left-biased  rollouts with life loss
            risk_right = fraction of right-biased rollouts with life loss
            risk_noop  = fraction of noop-biased  rollouts with life loss
            overall    = mean across all N rollouts
        """
        t0          = time.time()
        ale_state   = clone_ale_state(train_env)
        start_lives = get_lives(train_env) or 3   # fallback if ALE lives unavailable

        group_hits = [0, 0, 0]   # life-loss counts per policy group

        for group_idx in range(3):
            for _ in range(self.group_size):
                group_hits[group_idx] += self._run_one_rollout(
                    ale_state, start_lives, group_idx
                )

        risk_left  = group_hits[0] / self.group_size
        risk_right = group_hits[1] / self.group_size
        risk_noop  = group_hits[2] / self.group_size
        overall = max(risk_left, risk_right, risk_noop)

        elapsed = time.time() - t0
        if elapsed > 2.0:
            print(
                f"[MCRisk] Warning: evaluation took {elapsed:.2f}s "
                f"({self.n_rollouts} rollouts × {self.horizon_steps} steps). "
                "Consider reducing --mc_risk_horizon_steps or --mc_risk_group_size."
            )

        return overall, risk_left, risk_right, risk_noop

    def close(self) -> None:
        self._env.close()


def should_rescue(obs: np.ndarray, info: dict, env: gym.Env,
                  random_rescue_prob: float = 0.0) -> bool:
    """
    Placeholder: always returns False.
    Set --offense_random_rescue_prob > 0 for random rescue diversity.
    """
    if random_rescue_prob > 0.0 and random.random() < random_rescue_prob:
        return True
    return False


def risk_fn(obs: np.ndarray, info: dict, env: gym.Env) -> float:
    """Placeholder: returns 0.0 (risk shaping disabled)."""
    return 0.0


def train_step(policy_net: DQN, target_net: DQN, optimizer: optim.Adam,
               buf: ReplayBuffer, batch_size: int,
               gamma: float, device: torch.device,
               grad_clip: float = 10.0) -> float:

    obs_b, act_b, rew_b, nobs_b, done_b = buf.sample(batch_size)

    obs_t  = torch.from_numpy(obs_b ).to(device).float() / 255.0   # [B,4,84,84]
    nobs_t = torch.from_numpy(nobs_b).to(device).float() / 255.0
    act_t  = torch.from_numpy(act_b ).to(device)                    # [B] int64
    rew_t  = torch.from_numpy(rew_b ).to(device)                    # [B] float32
    done_t = torch.from_numpy(done_b).to(device)                    # [B] float32

    with torch.no_grad():
        next_q   = target_net(nobs_t).max(dim=1)[0]
        target_q = rew_t + gamma * next_q * (1.0 - done_t)

    current_q = policy_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
    loss      = F.huber_loss(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
    optimizer.step()

    return loss.item()


def save_checkpoint(policy_net: DQN, target_net: DQN,
                    optimizer: optim.Adam,
                    env_frames: int, decision_steps: int,
                    args, tag: str) -> None:
    path = os.path.join(args.save_dir, f"{args.mode}_dqn_{tag}.pt")
    torch.save({
        'policy_net'    : policy_net.state_dict(),
        'target_net'    : target_net.state_dict(),
        'optimizer'     : optimizer.state_dict(),
        'env_frames'    : env_frames,
        'decision_steps': decision_steps,
        'args'          : vars(args),
    }, path)
    print(f"[Checkpoint] Saved → {path}")


def run_training(args) -> None:

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Environment ───────────────────────────────────────────────────────────
    env       = make_env(args.env_id, seed=args.seed)
    n_actions = env.action_space.n
    print(f"[Env]  {args.env_id}  |  n_actions={n_actions}  |  device={device}")


    FRAME_SKIP = 4

    offense_intervention_steps: int = args.offense_intervention_frames // FRAME_SKIP
    defense_intervention_steps: int = args.defense_intervention_frames // FRAME_SKIP
    defense_period_steps: int       = args.defense_period_frames       // FRAME_SKIP

    print(
        f"[Config] mode={args.mode}\n"
        f"         offense intervention : {args.offense_intervention_frames} frames "
        f"= {offense_intervention_steps} steps\n"
        f"         defense intervention : {args.defense_intervention_frames} frames "
        f"= {defense_intervention_steps} steps\n"
        f"         defense period       : {args.defense_period_frames} frames "
        f"= {defense_period_steps} steps"
    )

    # ── Networks ─────────────────────────────────────────────────────────────
    policy_net = DQN(n_actions=n_actions).to(device)
    target_net = DQN(n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr, eps=1e-5)

    # ── Expert policy ─────────────────────────────────────────────────────────
    #
    #   The expert acts greedily (ε = 0) and runs in eval mode with no grad.
    #   Its transitions are NEVER inserted into the learner's replay buffer.
    expert_net = None
    if args.expert_checkpoint:
        expert_net = load_expert(args.expert_checkpoint, n_actions, device)
    else:
        print("[Expert] No checkpoint provided — expert intervention disabled.")

    mc_risk_detector = None
    mc_risk_eval_interval_steps = 0

    if args.mode == 'offense' and args.use_mc_risk_detector:
        mc_risk_detector = MCRiskDetector(
            env_id                = args.env_id,
            horizon_steps         = args.mc_risk_horizon_steps,
            frame_skip            = args.mc_risk_frame_skip,
            screen_size           = 84,
            n_stack               = 4,
            num_workers           = args.mc_risk_num_workers,
            seed                  = args.seed,
            group_size            = args.mc_risk_group_size,
            hit_life_delay_frames = args.mc_risk_hit_life_delay_frames,
        )
        mc_risk_eval_interval_steps = args.mc_risk_eval_interval_frames // FRAME_SKIP
        print(
            f"[MCRisk] Enabled: "
            f"N={3 * args.mc_risk_group_size} rollouts × "
            f"H={args.mc_risk_horizon_steps} steps | "
            f"eval every {mc_risk_eval_interval_steps} decision steps | "
            f"threshold={args.mc_risk_threshold}"
        )

    # Cached MC risk values (refreshed every mc_risk_eval_interval_steps steps).
    cached_mc_risk       = 0.0
    cached_mc_risk_left  = 0.0
    cached_mc_risk_right = 0.0
    cached_mc_risk_noop  = 0.0
    next_mc_risk_eval_step = 0   # evaluate on the very first eligible step

    # MC risk logging counters.
    total_mc_risk_evals = 0
    total_mc_risk_sum   = 0.0
    total_mc_rescues    = 0

    # ── Replay buffer ─────────────────────────────────────────────────────────
    replay_buf = ReplayBuffer(args.buffer_size)

    # ── Initial environment state ─────────────────────────────────────────────
    obs, info  = env.reset(seed=args.seed)
    prev_lives = get_lives(env)
    prev_risk  = risk_fn(obs, info, env)   # only used in defense mode

    # ── Training counters ─────────────────────────────────────────────────────
    env_frames     = 0
    decision_steps = 0

    # ── Episode trackers ──────────────────────────────────────────────────────
    ep_return_custom   = 0.0
    ep_return_original = 0.0
    ep_num             = 0

    # ── Expert intervention state machine ─────────────────────────────────────
    #
    #   expert_steps_remaining   > 0 → expert controls the current step
    #   cooldown_steps_remaining > 0 → learner acts but transitions are skipped
    #
    #   Transitions:
    #     LEARNER  → EXPERT   : rescue / periodic takeover triggered
    #     EXPERT   → COOLDOWN : expert_steps_remaining reaches 0
    #     COOLDOWN → LEARNER  : cooldown_steps_remaining reaches 0
    expert_steps_remaining   = 0
    cooldown_steps_remaining = 0

    # Defense: decision step at which the next periodic takeover is due.
    next_defense_takeover_step: int = defense_period_steps

    # ── Logging ───────────────────────────────────────────────────────────────
    total_expert_steps        = 0
    total_skipped_transitions = 0
    total_interventions       = 0
    recent_losses: deque      = deque(maxlen=200)

    next_checkpoint_frame = args.checkpoint_interval
    start_time            = time.time()

    print(f"[Train] Starting — total_env_frames={args.total_env_frames:,}")

    while env_frames < args.total_env_frames:

        epsilon = epsilon_schedule(env_frames, args)

        # ─────────────────────────────────────────────────────────────────────
        # Determine expert control for this decision step
        # ─────────────────────────────────────────────────────────────────────

        if args.mode == 'offense':
            # ── Offense: MC-guided or random rescue ──────────────────────────
            #
            #   HOW OFFENSE INTERVENTION WORKS
            #   ───────────────────────────────
            #   Every mc_risk_eval_interval_steps decision steps (when the learner
            #   is in control), the MC risk detector clones the ALE state and
            #   runs N biased rollouts to estimate death probability.  The result
            #   is cached so the per-step overhead is only one dict lookup.
            #
            #   If cached_mc_risk > mc_risk_threshold (or should_rescue() returns
            #   True in fallback mode), an expert rescue is triggered with
            #   probability offense_intervention_prob.
            #
            #   Expert transitions are excluded from the replay buffer because
            #   they are off-policy demonstrations from a different policy.
            #   The post-intervention cooldown (cooldown_steps_remaining) also
            #   excludes the few learner steps immediately after the expert
            #   returns control, since those states are still biased toward the
            #   unusually safe position left by the expert.

            if expert_steps_remaining > 0:
                expert_controlled       = True
                expert_steps_remaining -= 1
                if expert_steps_remaining == 0:
                    cooldown_steps_remaining = args.post_intervention_cooldown_steps
            else:
                expert_controlled = False

                # ── Refresh MC risk cache if due ──────────────────────────────
                #   Only evaluate risk when the learner is in control (no point
                #   evaluating during cooldown or expert intervention).
                if (mc_risk_detector is not None
                        and decision_steps >= next_mc_risk_eval_step):
                    (cached_mc_risk,
                     cached_mc_risk_left,
                     cached_mc_risk_right,
                     cached_mc_risk_noop) = mc_risk_detector.estimate_risk_from_env(env)
                    next_mc_risk_eval_step = decision_steps + mc_risk_eval_interval_steps
                    total_mc_risk_evals  += 1
                    total_mc_risk_sum    += cached_mc_risk

                # ── Rescue decision ───────────────────────────────────────────
                if args.use_mc_risk_detector:
                    do_rescue = cached_mc_risk > args.mc_risk_threshold
                else:
                    do_rescue = should_rescue(
                        obs, info, env,
                        random_rescue_prob=args.offense_random_rescue_prob,
                    )

                if expert_net is not None and do_rescue:
                    if random.random() < args.offense_intervention_prob:
                        expert_controlled      = True
                        expert_steps_remaining = offense_intervention_steps - 1
                        total_interventions   += 1
                        if args.use_mc_risk_detector:
                            total_mc_rescues += 1

        else:
            # ── Defense: periodic expert takeover ────────────────────────────
            #
            #   HOW DEFENSE INTERVENTION WORKS
            #   ───────────────────────────────
            #   Every defense_period_steps decision steps the expert takes over
            #   for defense_intervention_steps steps, providing diverse survival
            #   trajectories and preventing policy collapse to a single corner.
            #   Expert transitions are excluded from the replay buffer for the
            #   same reason as in offense mode.

            if expert_steps_remaining > 0:
                expert_controlled       = True
                expert_steps_remaining -= 1
                if expert_steps_remaining == 0:
                    cooldown_steps_remaining   = args.post_intervention_cooldown_steps
                    next_defense_takeover_step = decision_steps + defense_period_steps
            else:
                expert_controlled = False
                if expert_net is not None and decision_steps >= next_defense_takeover_step:
                    expert_controlled      = True
                    expert_steps_remaining = defense_intervention_steps - 1
                    total_interventions   += 1

        # ─────────────────────────────────────────────────────────────────────
        # Select action
        # ─────────────────────────────────────────────────────────────────────

        obs_t = obs_to_tensor(obs, device)   # (1, 4, 84, 84) float32 in [0,1]

        if expert_controlled and expert_net is not None:
            action = greedy_action(expert_net, obs_t)
            total_expert_steps += 1
        else:
            action = epsilon_greedy_action(policy_net, obs_t, epsilon, n_actions)

        # ─────────────────────────────────────────────────────────────────────
        # Step the environment
        # ─────────────────────────────────────────────────────────────────────

        next_obs, original_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ─────────────────────────────────────────────────────────────────────
        # Life-loss detection
        # ─────────────────────────────────────────────────────────────────────

        current_lives = get_lives(env)
        life_lost = (
            prev_lives is not None
            and current_lives is not None
            and current_lives < prev_lives
        )

        # ─────────────────────────────────────────────────────────────────────
        # Compute mode-specific reward
        # ─────────────────────────────────────────────────────────────────────

        if args.mode == 'offense':
            # Raw game score; add invasion penalty on terminal if enabled.
            custom_reward = float(original_reward)
            if terminated and args.invasion_penalty != 0.0:
                try:
                    ram = env.unwrapped.ale.getRAM()
                    # RAM[0x78]==84 → 3 bullet hits (bullet death)
                    # RAM[0x78]!=84 → enemies reached bottom (invasion)
                    if ram[0x78] != 84:
                        custom_reward -= args.invasion_penalty
                except Exception:
                    pass

        else:  # defense
            # Score is removed entirely; shape purely for survival.
            #   defense_reward = survival_reward                 (alive bonus)
            #                  - death_penalty  if life_lost     (life-loss penalty)
            #                  + alpha × (prev_risk - current_risk)
            current_risk  = risk_fn(next_obs, info, env)
            custom_reward = args.survival_reward
            if life_lost:
                custom_reward -= args.death_penalty
            custom_reward += args.risk_reduction_alpha * (prev_risk - current_risk)
            prev_risk      = current_risk

        # ─────────────────────────────────────────────────────────────────────
        # Decide whether to store the transition
        # ─────────────────────────────────────────────────────────────────────
        #
        # Store only if the LEARNER selected this action AND we are not in the
        # post-intervention cooldown window.
        #
        # WHY EXPERT-CONTROLLED FRAMES ARE EXCLUDED
        # ──────────────────────────────────────────
        # Expert transitions are generated by a different (typically better)
        # policy. Storing them would bias the learner toward the expert's
        # strategy rather than letting it develop its own.  The learner still
        # observes next_obs after every step, so its state estimate stays accurate.
        #
        # WHY THE COOLDOWN WINDOW IS EXCLUDED
        # ────────────────────────────────────
        # States immediately following expert takeovers are biased toward
        # unusually safe configurations that the learner did not navigate to on
        # its own.  Excluding the first post_intervention_cooldown_steps learner
        # transitions removes this bias from the replay buffer.

        if expert_controlled:
            store_transition = False
            total_skipped_transitions += 1
        elif cooldown_steps_remaining > 0:
            store_transition          = False
            cooldown_steps_remaining -= 1
            total_skipped_transitions += 1
        else:
            store_transition = True

        if store_transition:
            replay_buf.push(
                np.array(obs,      dtype=np.uint8),
                action,
                custom_reward,
                np.array(next_obs, dtype=np.uint8),
                float(done),
            )

        # ─────────────────────────────────────────────────────────────────────
        # Episode statistics
        # ─────────────────────────────────────────────────────────────────────

        ep_return_custom   += custom_reward
        ep_return_original += float(original_reward)

        # ─────────────────────────────────────────────────────────────────────
        # Train the learner
        # ─────────────────────────────────────────────────────────────────────

        if (env_frames >= args.learning_starts
                and len(replay_buf) >= args.batch_size
                and decision_steps % args.train_frequency == 0):
            loss_val = train_step(
                policy_net, target_net, optimizer,
                replay_buf, args.batch_size, args.gamma, device,
            )
            recent_losses.append(loss_val)

        # ─────────────────────────────────────────────────────────────────────
        # Update target network
        # ─────────────────────────────────────────────────────────────────────

        if decision_steps % args.target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # ─────────────────────────────────────────────────────────────────────
        # Advance counters
        # ─────────────────────────────────────────────────────────────────────

        prev_lives     = current_lives
        obs            = next_obs
        env_frames    += FRAME_SKIP
        decision_steps += 1

        # ─────────────────────────────────────────────────────────────────────
        # Handle episode end
        # ─────────────────────────────────────────────────────────────────────

        if done:
            ep_num += 1

            if ep_num % 10 == 0:
                elapsed    = time.time() - start_time
                fps        = env_frames / max(elapsed, 1e-6)
                mean_loss  = float(np.mean(recent_losses)) if recent_losses else float('nan')
                expert_pct = 100.0 * total_expert_steps / max(1, decision_steps)
                avg_mc     = total_mc_risk_sum / max(1, total_mc_risk_evals)

                mc_str = ""
                if args.use_mc_risk_detector:
                    mc_str = (
                        f"  mc={cached_mc_risk:.3f}"
                        f"(L={cached_mc_risk_left:.2f}"
                        f"/R={cached_mc_risk_right:.2f}"
                        f"/N={cached_mc_risk_noop:.2f})"
                        f"  mc_evals={total_mc_risk_evals}"
                        f"  mc_avg={avg_mc:.3f}"
                        f"  mc_rescues={total_mc_rescues}"
                    )

                print(
                    f"[Ep {ep_num:5d}] "
                    f"frames={env_frames:9,}  "
                    f"ε={epsilon:.3f}  "
                    f"ret_custom={ep_return_custom:8.1f}  "
                    f"ret_orig={ep_return_original:8.1f}  "
                    f"loss={mean_loss:.4f}  "
                    f"buf={len(replay_buf):7,}  "
                    f"expert={expert_pct:4.1f}%  "
                    f"interventions={total_interventions:5,}  "
                    f"skipped={total_skipped_transitions:7,}  "
                    f"fps={fps:5.0f}"
                    + mc_str
                )

            # Reset episode accumulators.
            ep_return_custom   = 0.0
            ep_return_original = 0.0

            # Clear intervention state — it belongs to the finished episode.
            expert_steps_remaining   = 0
            cooldown_steps_remaining = 0

            obs, info  = env.reset(seed=args.seed + ep_num)
            prev_lives = get_lives(env)
            prev_risk  = risk_fn(obs, info, env)

        # ─────────────────────────────────────────────────────────────────────
        # Periodic checkpoint
        # ─────────────────────────────────────────────────────────────────────

        if env_frames >= next_checkpoint_frame:
            save_checkpoint(
                policy_net, target_net, optimizer,
                env_frames, decision_steps, args,
                tag=f"frames{env_frames}",
            )
            next_checkpoint_frame += args.checkpoint_interval

    # ── Training complete ─────────────────────────────────────────────────────
    env.close()
    if mc_risk_detector is not None:
        mc_risk_detector.close()

    save_checkpoint(
        policy_net, target_net, optimizer,
        env_frames, decision_steps, args,
        tag="final",
    )
    elapsed = time.time() - start_time
    print(
        f"[Done] {env_frames:,} env frames | {ep_num} episodes | "
        f"{elapsed/3600:.2f}h elapsed"
    )


# =============================================================================
# 12.  Argument parser
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Offense / Defense DQN for Atari Space Invaders',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Mode & environment ────────────────────────────────────────────────────
    p.add_argument('--mode',
                   required=True, choices=['offense', 'defense'])
    p.add_argument('--env_id',
                   default='SpaceInvadersNoFrameskip-v4')
    p.add_argument('--expert_checkpoint',
                   default='pretrained/dqn_cnn.pt')
    p.add_argument('--save_dir',
                   default='checkpoints')

    # ── Training budget ───────────────────────────────────────────────────────
    p.add_argument('--total_env_frames',   type=int,   default=10_000_000)
    p.add_argument('--learning_starts',    type=int,   default=50_000,
                   help='Env frames before gradient steps begin')
    p.add_argument('--buffer_size',        type=int,   default=500_000)
    p.add_argument('--batch_size',         type=int,   default=32)
    p.add_argument('--gamma',              type=float, default=0.99)
    p.add_argument('--lr',                 type=float, default=1e-4)
    p.add_argument('--target_update_interval', type=int, default=1000,
                   help='Decision steps between target network syncs')
    p.add_argument('--train_frequency',    type=int,   default=4,
                   help='Gradient step every N decision steps')

    # ── Epsilon-greedy schedule ───────────────────────────────────────────────
    p.add_argument('--epsilon_start',        type=float, default=1.0)
    p.add_argument('--epsilon_final',        type=float, default=0.01)
    p.add_argument('--epsilon_decay_frames', type=int,   default=1_000_000)

    # ── Hardware & reproducibility ────────────────────────────────────────────
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',   type=int, default=42)

    # ── Intervention (shared) ─────────────────────────────────────────────────
    p.add_argument('--post_intervention_cooldown_steps', type=int, default=4,
                   help='Learner decision steps to skip from replay after expert ends')

    # ── Offense intervention ──────────────────────────────────────────────────
    p.add_argument('--offense_intervention_prob',   type=float, default=0.8,
                   help='Probability of rescuing when rescue condition is met')
    p.add_argument('--offense_intervention_frames', type=int,   default=40,
                   help='Env frames per rescue (40 frames = 10 decision steps)')
    p.add_argument('--offense_random_rescue_prob',  type=float, default=0.0,
                   help='Fallback: random rescue probability when MC detector is off')

    # ── Defense intervention ──────────────────────────────────────────────────
    p.add_argument('--defense_period_frames',       type=int,   default=400,
                   help='Env frames between expert takeovers (400 = 100 steps)')
    p.add_argument('--defense_intervention_frames', type=int,   default=80,
                   help='Env frames per expert takeover (80 = 20 steps)')

    # ── Offense reward ────────────────────────────────────────────────────────
    p.add_argument('--invasion_penalty',     type=float, default=100.0,
                   help='Penalty subtracted from reward when episode ends by '
                        'enemy invasion (RAM[0x78]!=84 at terminal). ')

    # ── Defense reward ────────────────────────────────────────────────────────
    p.add_argument('--survival_reward',      type=float, default=0.001)
    p.add_argument('--death_penalty',        type=float, default=1.0)
    p.add_argument('--risk_reduction_alpha', type=float, default=1.0,
                   help='Weight for risk-reduction bonus (0 = disabled)')

    # ── MC biased-rollout risk detector ──────────────────────────────────────
    p.add_argument('--use_mc_risk_detector', action='store_true',
                   help='Enable MC biased-rollout death-risk detector for offense')
    p.add_argument('--mc_risk_horizon_steps', type=int, default=13,
                   help='Action probe horizon in decision steps '
                        '(action_window_frames = horizon_steps × frame_skip; '
                        'default 13 → 52 raw frames of biased actions). '
                        'Total rollout = hit_life_delay + action_window raw frames.')
    p.add_argument('--mc_risk_eval_interval_frames', type=int, default=80,
                   help='Raw env frames between MC risk evaluations '
                        '(80 frames = 20 decision steps)')
    p.add_argument('--mc_risk_threshold', type=float, default=0.5,
                   help='Risk score above which expert rescue is triggered')
    p.add_argument('--mc_risk_num_workers', type=int, default=0,
                   help='Parallel rollout workers (0 = sequential; >0 reserved)')
    p.add_argument('--mc_risk_frame_skip', type=int, default=4,
                   help='Raw frames per decision step inside MC rollouts')
    p.add_argument('--mc_risk_group_size', type=int, default=8,
                   help='Rollouts per policy group; total N = 3 × group_size')
    p.add_argument('--mc_risk_hit_life_delay_frames', type=int, default=120,
                   help='Raw frames between a bullet hit and the ALE life-count decrease '
                        '(~120 for Space Invaders); sets risk_min_frame and the length of '
                        'the Phase-2 NOOP-wait. Life losses before this are treated as '
                        'pre-existing hits and do NOT count as risk events.')

    # ── Checkpointing ─────────────────────────────────────────────────────────
    p.add_argument('--checkpoint_interval', type=int, default=1_000_000,
                   help='Env frames between periodic checkpoint saves')

    return p.parse_args()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    args = parse_args()
    run_training(args)
