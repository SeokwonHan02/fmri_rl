#!/usr/bin/env python3
"""
find_hit_ram.py — Find Space Invaders RAM bytes that change transiently
when the player is HIT by a bullet (life-loss event).

Strategy:
  1. Play raw (frame_skip=1) episodes with DQN.
  2. Detect each life-loss moment: the frame where ale.lives() decrements.
  3. Record RAM for a window [-pre_frames, +post_frames] around each hit.
  4. Compare RAM at hit frames vs. RAM at random non-hit frames.
  5. Also look at the window BEFORE the life decrement for a transient "hit"
     flag that fires at bullet contact (before the death animation finishes
     and lives actually tick down).

Invasion detection logic (once the byte is found):
  - Count hits during episode = number of times the hit byte fires.
  - If terminated=True and hit_count < 3 → invasion.
  - If terminated=True and hit_count >= 3 → bullet death on last life.

Output: hit_ram_analysis.png + console report of candidate bytes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass


# ── DQN ───────────────────────────────────────────────────────────────────────

class DQN(nn.Module):
    def __init__(self, n_actions=6):
        super().__init__()
        self.conv1  = nn.Conv2d(4,  32, kernel_size=8, stride=4)
        self.conv2  = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3  = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc3    = nn.Linear(3136, 512)
        self.fc_out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc3(x))
        return self.fc_out(x)


# ── Env ───────────────────────────────────────────────────────────────────────

def make_raw_env(seed=0):
    """Raw env (frame_skip=1) — needed to call setRAM/getRAM every ALE step."""
    env = gym.make('SpaceInvadersNoFrameskip-v4', render_mode='rgb_array')
    env = AtariPreprocessing(
        env, noop_max=0, frame_skip=1, screen_size=84,
        terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False,
    )
    env.reset(seed=seed)
    return env


def make_stacked_env(seed=0):
    """fs=4 stacked env for DQN inference; ale handle returned separately."""
    env = gym.make('SpaceInvadersNoFrameskip-v4', render_mode='rgb_array')
    env = AtariPreprocessing(
        env, noop_max=0, frame_skip=4, screen_size=84,
        terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    env.reset(seed=seed)
    return env


# ── Data collection ───────────────────────────────────────────────────────────

def collect_hit_windows(
    n_episodes=30,
    checkpoint='pretrained/dqn_cnn.pt',
    seed=42,
    epsilon=0.05,
    pre_frames=10,   # raw frames before life decrement
    post_frames=10,  # raw frames after life decrement
):
    """
    Play raw env with DQN (inferred every 4 raw frames).
    At each life-loss frame record a window of RAM snapshots.
    Also collect random non-hit RAM snapshots for comparison.

    Returns:
      hit_windows  : list of np.array (pre+1+post, 128) — centered on hit frame
      random_rams  : list of np.array (128,) — RAM at random non-hit frames
      hit_frame_rams : np.array (N_hits, 128) — RAM exactly at the hit frame
    """
    device = torch.device('cpu')

    # Load DQN for inference
    net_env = make_stacked_env(seed=seed)  # for obs stacking
    net = DQN(n_actions=6).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt.get('policy_net', ckpt))
    net.eval()
    net_env.close()

    # Raw env for step-level RAM access
    env = make_raw_env(seed=seed)
    ale = env.unwrapped.ale

    # We need 4-frame obs history for DQN input
    obs_buf = []   # ring buffer of last 4 grayscale 84×84 frames

    hit_windows  = []
    random_rams  = []
    ram_ring     = []   # circular buffer of last (pre_frames+1) RAMs

    def get_stacked_obs():
        if len(obs_buf) < 4:
            frames = [obs_buf[0]] * (4 - len(obs_buf)) + list(obs_buf)
        else:
            frames = list(obs_buf)
        return np.stack(frames, axis=0).astype(np.uint8)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_buf.clear()
        obs_buf.append(obs)
        ram_ring.clear()

        prev_lives = ale.lives()
        step       = 0
        action     = 0
        done       = False

        while not done:
            ram = ale.getRAM().copy()
            ram_ring.append(ram)
            if len(ram_ring) > pre_frames + 1:
                ram_ring.pop(0)

            cur_lives = ale.lives()

            # ── Life-loss detected ──────────────────────────────────────────
            if cur_lives < prev_lives:
                # Collect post-hit frames
                post_rams = [ram]
                for _ in range(post_frames):
                    obs_next, _, term, trunc, _ = env.step(0)  # NOOP
                    obs_buf.append(obs_next[-1] if obs_next.ndim > 2 else obs_next)
                    if len(obs_buf) > 4:
                        obs_buf.pop(0)
                    post_rams.append(ale.getRAM().copy())
                    if term or trunc:
                        done = True
                        break

                # Build window: pre + hit + post
                pre_part  = list(ram_ring)  # already has current ram at end
                post_part = post_rams[1:]   # skip duplicate of hit frame
                window    = pre_part + post_part
                # Pad if needed
                while len(window) < pre_frames + 1 + post_frames:
                    window = [window[0]] + window
                window = window[-(pre_frames + 1 + post_frames):]
                hit_windows.append(np.array(window, dtype=np.uint8))
                prev_lives = cur_lives
                step += 1
                if not done:
                    obs, _, term, trunc, _ = env.step(action)
                    obs_buf.append(obs[-1] if obs.ndim > 2 else obs)
                    if len(obs_buf) > 4:
                        obs_buf.pop(0)
                    done = term or trunc
                continue

            # ── Random non-hit frame ────────────────────────────────────────
            if step % 50 == 25:
                random_rams.append(ram.copy())

            # ── DQN action every 4 raw frames ──────────────────────────────
            if step % 4 == 0:
                stacked = get_stacked_obs()
                t = torch.from_numpy(stacked).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    q = net(t)
                action = (env.action_space.sample()
                          if np.random.random() < epsilon
                          else q.argmax(dim=1).item())

            obs, _, term, trunc, _ = env.step(action)
            obs_buf.append(obs[-1] if obs.ndim > 2 else obs)
            if len(obs_buf) > 4:
                obs_buf.pop(0)

            prev_lives = cur_lives
            step += 1
            done = term or trunc

        n_hits = len(hit_windows)
        print(f"  Ep {ep+1:3d}: collected {n_hits} hit events so far  "
              f"(lives at end={ale.lives()})")

    env.close()

    hit_frame_rams = np.array([w[pre_frames] for w in hit_windows], dtype=np.uint8) \
                     if hit_windows else np.zeros((0, 128), dtype=np.uint8)
    random_array   = np.array(random_rams, dtype=np.uint8) \
                     if random_rams else np.zeros((0, 128), dtype=np.uint8)
    return hit_windows, random_array, hit_frame_rams


# ── Analysis ──────────────────────────────────────────────────────────────────

def cohen_d(a, b):
    """Cohen's d between two 1-D arrays."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    var_pool = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    std_pool = np.sqrt(var_pool) if var_pool > 0 else 1e-9
    return abs(float(a.mean()) - float(b.mean())) / std_pool


def analyse_hits(hit_windows, random_rams, pre_frames, post_frames, top_k=20):
    """
    For each of the 128 RAM bytes, compute:
      1. Mean value at hit frame vs random frames → Cohen's d (hit discriminator)
      2. Mean temporal profile across the window (pre + hit + post) to see
         transient changes.

    Also look at the PRE-hit window for bytes that fire BEFORE the life counter
    decrements (these would be the earliest detectable "hit" signal).
    """
    if len(hit_windows) == 0:
        print("No hit events collected!")
        return None, None, None

    windows = np.array(hit_windows, dtype=float)  # (N, pre+1+post, 128)
    N       = windows.shape[0]
    hit_idx = pre_frames  # center frame index

    hit_rams    = windows[:, hit_idx, :]          # (N, 128)
    pre_rams    = windows[:, :hit_idx, :]          # (N, pre, 128)
    rnd         = random_rams.astype(float)        # (M, 128)

    # Cohen's d: hit frame vs random
    d_hit   = np.array([cohen_d(hit_rams[:, b], rnd[:, b]) for b in range(128)])

    # Look for bytes that change BEFORE the hit frame (transient hit flag)
    baseline  = rnd.mean(axis=0)   # (128,)
    pre_delta = np.abs(pre_rams - baseline[np.newaxis, np.newaxis, :]).mean(axis=(0, 1))

    # Per-frame Cohen's d in the pre-hit window: shape (pre_frames, 128)
    pre_d_per_frame = np.array([
        [cohen_d(windows[:, f, b], rnd[:, b]) for b in range(128)]
        for f in range(hit_idx)
    ])  # (pre_frames, 128)

    # Earliest frame (relative to life decrement) where d >= threshold
    EARLY_THRESH = 1.0
    earliest_frame = {}   # addr -> relative frame (negative = before decrement)
    for b in range(128):
        frames_above = np.where(pre_d_per_frame[:, b] >= EARLY_THRESH)[0]
        if len(frames_above) > 0:
            earliest_frame[b] = frames_above[0] - hit_idx  # negative

    # Also: bytes that transition at exactly pre-1 frame (1 frame before life drops)
    if hit_idx >= 1:
        d_pre1 = np.array([cohen_d(windows[:, hit_idx-1, b], rnd[:, b])
                           for b in range(128)])
    else:
        d_pre1 = np.zeros(128)

    # Rank by d_hit
    top_idx = np.argsort(d_hit)[::-1][:top_k]

    print(f"\n{'='*70}")
    print(f"Hit-frame RAM discriminators  (hit n={N}, random n={len(rnd)})")
    print(f"{'='*70}")
    print(f"{'Addr':>6}  {'Cohen d':>9}  {'Hit mean':>9}  {'Rnd mean':>9}  "
          f"{'Pre-1 d':>9}  {'PreΔ':>9}")
    print("-" * 70)
    for b in top_idx:
        print(f"0x{b:02X}={b:3d}  {d_hit[b]:9.3f}  {hit_rams[:,b].mean():9.2f}  "
              f"{rnd[:,b].mean():9.2f}  {d_pre1[b]:9.3f}  {pre_delta[b]:9.3f}")

    # Bytes that fire BEFORE the hit frame — ranked by how EARLY they fire
    print(f"\n{'='*70}")
    print(f"EARLY-HIT candidates  (d>={EARLY_THRESH} threshold, ranked by earliest frame)")
    print(f"  frame is relative to life decrement: -120 means 120 raw frames BEFORE lives drop")
    print(f"{'='*70}")
    print(f"{'Addr':>6}  {'EarliestF':>10}  {'PreΔ':>9}  {'d_hit':>9}  "
          f"{'hit_mean':>9}  {'rnd_mean':>9}")
    print("-" * 70)
    # Sort by earliest frame (most negative = fires earliest)
    sorted_early = sorted(earliest_frame.items(), key=lambda x: x[1])
    for b, ef in sorted_early[:top_k]:
        print(f"0x{b:02X}={b:3d}  {ef:10d}  {pre_delta[b]:9.3f}  {d_hit[b]:9.3f}  "
              f"{hit_rams[:,b].mean():9.2f}  {rnd[:,b].mean():9.2f}")

    if not sorted_early:
        print("  (no bytes exceeded the threshold in the pre-hit window)")

    # Also show per-frame d profile for the top early candidates
    print(f"\n{'='*70}")
    print("Per-frame Cohen's d profile (10-frame bins) for top-5 early candidates")
    print(f"{'='*70}")
    top5_early = [b for b, _ in sorted_early[:5]]
    bin_size = max(1, hit_idx // 16)
    header = "Addr      " + "".join(f"{-(hit_idx - f*bin_size):>6d}" for f in range(hit_idx // bin_size))
    print(header[:120])
    for b in top5_early:
        row = f"0x{b:02X}={b:3d} "
        for f0 in range(0, hit_idx, bin_size):
            chunk = pre_d_per_frame[f0:f0+bin_size, b]
            row += f"{chunk.mean():6.2f}"
        print(row[:120])

    return d_hit, pre_delta, windows, earliest_frame, pre_d_per_frame


# ── Verification: does the hit byte track cumulative hits? ────────────────────

def verify_hit_byte(addr, n_episodes=10, checkpoint='pretrained/dqn_cnn.pt',
                    seed=99, epsilon=0.05):
    """
    Check whether ram[addr] increments monotonically across hits in a full episode.
    Prints each hit moment with ram[addr] value before/after.
    """
    device = torch.device('cpu')
    net_env = make_stacked_env(seed=seed)
    net = DQN(n_actions=6).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt.get('policy_net', ckpt))
    net.eval()
    net_env.close()

    env = make_raw_env(seed=seed)
    ale = env.unwrapped.ale
    obs_buf = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_buf.clear()
        obs_buf.append(obs)
        prev_lives = ale.lives()
        step = 0
        action = 0
        done = False
        hit_count = 0

        print(f"\nEp {ep+1}: addr=0x{addr:02X}  initial_ram={ale.getRAM()[addr]}")

        while not done:
            cur_lives = ale.lives()
            if cur_lives < prev_lives:
                r = ale.getRAM()
                hit_count += 1
                print(f"  HIT {hit_count}: step={step}  lives {prev_lives}→{cur_lives}  "
                      f"ram[0x{addr:02X}]={r[addr]}")
                prev_lives = cur_lives

            if step % 4 == 0:
                if len(obs_buf) < 4:
                    stk = [obs_buf[0]] * (4 - len(obs_buf)) + list(obs_buf)
                else:
                    stk = list(obs_buf)
                stk = np.stack(stk, axis=0).astype(np.uint8)
                t = torch.from_numpy(stk).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    q = net(t)
                action = (env.action_space.sample()
                          if np.random.random() < epsilon
                          else q.argmax(dim=1).item())

            obs, _, term, trunc, _ = env.step(action)
            obs_buf.append(obs[-1] if obs.ndim > 2 else obs)
            if len(obs_buf) > 4:
                obs_buf.pop(0)
            step += 1
            done = term or trunc

        r_final = ale.getRAM()
        print(f"  END: total_hits={hit_count}  "
              f"ram[0x{addr:02X}]={r_final[addr]}  lives={ale.lives()}")

    env.close()


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(d_hit, pre_delta, windows, pre_frames, post_frames, top_k=8,
              out='hit_ram_analysis.png'):
    if windows is None:
        return
    N    = windows.shape[0]
    t    = np.arange(-(pre_frames), post_frames + 1)   # relative frame index

    top_d  = np.argsort(d_hit)[::-1][:top_k]
    top_pd = np.argsort(pre_delta)[::-1][:top_k]

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: d_hit bar chart ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    vals = d_hit[top_d]
    ax1.barh([f'0x{b:02X}' for b in top_d[::-1]], vals[::-1], color='steelblue')
    ax1.set_xlabel("Cohen's d  (hit frame vs random)")
    ax1.set_title("Top bytes at HIT frame")
    ax1.axvline(0.8, color='red', ls='--', lw=0.8, label='d=0.8')
    ax1.legend(fontsize=7)

    # ── Panel 2: pre_delta bar chart ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    vals2 = pre_delta[top_pd]
    ax2.barh([f'0x{b:02X}' for b in top_pd[::-1]], vals2[::-1], color='darkorange')
    ax2.set_xlabel("Mean |Δ| in pre-hit window")
    ax2.set_title("Top bytes in PRE-HIT window (early detection)")
    ax2.axvline(1.0, color='red', ls='--', lw=0.8)

    # ── Panel 3: temporal profiles of top d_hit bytes ────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    mean_traj = windows.mean(axis=0)    # (T, 128)
    std_traj  = windows.std(axis=0)
    cmap = plt.cm.tab10
    for i, b in enumerate(top_d[:6]):
        mu  = mean_traj[:, b]
        sd  = std_traj[:, b]
        col = cmap(i)
        ax3.plot(t, mu, label=f'0x{b:02X}', color=col)
        ax3.fill_between(t, mu - sd, mu + sd, alpha=0.15, color=col)
    ax3.axvline(0, color='black', ls='--', lw=1.2, label='life decrement')
    ax3.set_xlabel("Frame relative to life-loss (0 = life decrements)")
    ax3.set_ylabel("RAM byte value")
    ax3.set_title(f"Temporal profiles around life-loss  (mean ± std, n={N})")
    ax3.legend(fontsize=7, ncol=4)

    # ── Panel 4: heatmap of top bytes over time ───────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    top_all = list(dict.fromkeys(list(top_d[:8]) + list(top_pd[:8])))[:12]
    mat = mean_traj[:, top_all].T   # (n_bytes, T)
    im  = ax4.imshow(mat, aspect='auto', cmap='RdYlBu_r',
                     extent=[t[0] - 0.5, t[-1] + 0.5, len(top_all) - 0.5, -0.5])
    ax4.set_yticks(range(len(top_all)))
    ax4.set_yticklabels([f'0x{b:02X}' for b in top_all], fontsize=7)
    ax4.axvline(0, color='white', lw=1.5, ls='--')
    ax4.set_xlabel("Frame relative to life-loss")
    ax4.set_title("RAM heatmap around hit  (mean value)")
    plt.colorbar(im, ax=ax4, shrink=0.8)

    # ── Panel 5: top-1 pre-hit byte detailed view ────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    b1 = top_pd[0]
    for i in range(min(N, 20)):
        ax5.plot(t, windows[i, :, b1], color='orange', alpha=0.3, lw=0.8)
    ax5.plot(t, mean_traj[:, b1], color='darkorange', lw=2, label='mean')
    ax5.axvline(0, color='black', ls='--', lw=1.2)
    ax5.set_xlabel("Frame relative to life-loss")
    ax5.set_ylabel("RAM byte value")
    ax5.set_title(f"0x{b1:02X} individual hit traces")
    ax5.legend(fontsize=7)

    fig.suptitle("Space Invaders RAM — Player Hit Detection Analysis", fontsize=13, y=1.01)
    plt.savefig(out, bbox_inches='tight', dpi=120)
    print(f"\nSaved: {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    PRE   = 160  # raw frames before life decrement (bullet contact ~120f before lives drop)
    POST  = 30   # raw frames after life decrement
    N_EP  = 40   # episodes to collect

    print("Collecting hit windows (raw env, DQN policy)...")
    hit_windows, random_rams, hit_frame_rams = collect_hit_windows(
        n_episodes   = N_EP,
        checkpoint   = 'pretrained/dqn_cnn.pt',
        seed         = 42,
        epsilon      = 0.10,
        pre_frames   = PRE,
        post_frames  = POST,
    )
    print(f"\nCollected {len(hit_windows)} hit events, "
          f"{len(random_rams)} random RAM snapshots")

    if len(hit_windows) == 0:
        print("No hits collected — check env setup.")
    else:
        d_hit, pre_delta, windows, earliest_frame, pre_d_per_frame = analyse_hits(
            hit_windows, random_rams, PRE, POST, top_k=20
        )

        visualize(d_hit, pre_delta, windows, PRE, POST,
                  top_k=8, out='hit_ram_analysis.png')

        # Verify top-3 early-hit candidates (if any found)
        if earliest_frame:
            top3_early = [b for b, _ in sorted(earliest_frame.items(), key=lambda x: x[1])[:3]]
            print("\n" + "="*65)
            print("Verifying top-3 EARLIEST-hit bytes across 5 fresh episodes")
            print("="*65)
            for b in top3_early:
                print(f"\n--- addr=0x{b:02X}  earliest_frame={earliest_frame[b]} ---")
                verify_hit_byte(b, n_episodes=5, checkpoint='pretrained/dqn_cnn.pt',
                                seed=200, epsilon=0.10)
