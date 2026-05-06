#!/usr/bin/env python3
"""
find_death_ram.py — Empirically discover Space Invaders RAM addresses that
discriminate bullet death vs enemy invasion, using the pretrained DQN.

Key insight (confirmed empirically):
  In ALE Space Invaders, game_over() fires only when RAM[0x49] (lives) == 0.
  Both death types result in lives == 0 at the terminal frame, so lives alone
  cannot distinguish them.

Collection strategy:
  BULLET  — normal DQN play; all terminal events are bullet deaths because
             invasion never occurs before the player dies 3 times.
  INVASION — setRAM(0x49, 3) every step keeps the player immortal; the only
             way game_over() can trigger is by the enemy invasion condition
             in the ROM (enemies reaching the player row).

Output: ram_death_analysis.png
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

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

def make_env(seed=0):
    """Standard training env with frame_skip=4 and 4-frame stack."""
    env = gym.make('SpaceInvadersNoFrameskip-v4', render_mode='rgb_array')
    env = AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84,
        terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    env.reset(seed=seed)
    return env


def make_raw_env(seed=0):
    """Raw env (frame_skip=1) for RAM-level analysis."""
    env = gym.make('SpaceInvadersNoFrameskip-v4', render_mode='rgb_array')
    env = AtariPreprocessing(
        env, noop_max=0, frame_skip=1, screen_size=84,
        terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False,
    )
    env.reset(seed=seed)
    return env


# ── Bullet death collection ───────────────────────────────────────────────────

def collect_bullet_events(n_episodes=60, checkpoint='pretrained/dqn_cnn.pt',
                          history_len=40, seed=42, epsilon=0.05):
    """
    Normal DQN play.  All terminal events are bullet deaths: invasion never
    occurs before the player exhausts all 3 lives.
    history_len is in decision steps (frame_skip=4 wrapped env).
    """
    device = torch.device('cpu')
    net = DQN(n_actions=6).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt.get('policy_net', ckpt))
    net.eval()

    env = make_env(seed=seed)
    ale = env.unwrapped.ale
    events = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ram_buf = []
        ep_return = 0.0
        done = False

        while not done:
            # Record RAM at the raw ALE level (frame_skip=4 wrapped, so 4 raw
            # frames have already advanced; this is the decision-step RAM)
            ram_buf.append(ale.getRAM().copy())
            if len(ram_buf) > history_len:
                ram_buf.pop(0)

            t = torch.from_numpy(
                    np.array(obs, dtype=np.uint8)
                ).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                q = net(t)
            action = (env.action_space.sample()
                      if np.random.random() < epsilon
                      else q.argmax(dim=1).item())

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated

            if terminated:
                rgb = env.render()
                ram_buf.append(ale.getRAM().copy())
                if len(ram_buf) > history_len:
                    ram_buf.pop(0)

                events.append(dict(
                    type        = 'bullet',
                    ram_history = np.array(ram_buf, dtype=np.uint8),
                    rgb         = rgb if rgb is not None
                                  else np.zeros((210, 160, 3), np.uint8),
                    score       = ep_return,
                ))
                print(f"  Ep {ep+1:3d}: [bullet  ]  score={ep_return:.0f}  "
                      f"total={len(events)}")

    env.close()
    return events


# ── Invasion collection ───────────────────────────────────────────────────────

def collect_invasion_events(n_events=60, history_len=40, seed=42):
    """
    Immortal player (lives pinned to 3 via setRAM every raw frame).
    The only game_over trigger left is the ROM's invasion condition
    (enemies reaching the player row).  frame_skip=1 raw env is used so
    we can call setRAM before every ALE step.
    history_len is in raw frames.
    """
    env = make_raw_env(seed=seed)
    ale = env.unwrapped.ale
    events = []
    ep = 0

    while len(events) < n_events:
        env.reset(seed=seed + ep)
        ep += 1
        ram_buf = []
        step = 0
        done = False

        while not done:
            ale.setRAM(0x49, 3)   # pin lives = 3 → bullets can never end the game

            ram_buf.append(ale.getRAM().copy())
            if len(ram_buf) > history_len:
                ram_buf.pop(0)

            _, _, terminated, truncated, _ = env.step(0)   # NOOP — no fire
            done = terminated or truncated
            step += 1

            if terminated:
                rgb = env.render()
                ram_buf.append(ale.getRAM().copy())
                if len(ram_buf) > history_len:
                    ram_buf.pop(0)

                events.append(dict(
                    type        = 'invasion',
                    ram_history = np.array(ram_buf, dtype=np.uint8),
                    rgb         = rgb if rgb is not None
                                  else np.zeros((210, 160, 3), np.uint8),
                    ep_steps    = step,
                ))
                print(f"  Ep {ep:3d}: [invasion]  steps={step}  total={len(events)}")

            if step > 20000:
                print(f"  Ep {ep:3d}: no terminal in 20000 steps, resetting")
                break

    env.close()
    return events


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(bullet_events, invasion_events, top_k=20):
    """Cohen's d per RAM byte at the terminal frame."""
    b_rams = np.array([e['ram_history'][-1] for e in bullet_events], dtype=float)
    i_rams = np.array([e['ram_history'][-1] for e in invasion_events], dtype=float)

    mean_b, std_b = b_rams.mean(0), b_rams.std(0)
    mean_i, std_i = i_rams.mean(0), i_rams.std(0)
    pooled        = np.sqrt((std_b**2 + std_i**2) / 2) + 1e-6
    d_score       = np.abs(mean_b - mean_i) / pooled
    const         = (std_b < 3.0) & (std_i < 3.0)

    return np.argsort(d_score)[::-1][:top_k], d_score, mean_b, mean_i, std_b, std_i, const


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize(bullet_events, invasion_events,
              sorted_idx, d_score, mean_b, mean_i, std_b, std_i, const,
              top_k=16):

    nb, ni = len(bullet_events), len(invasion_events)
    top    = sorted_idx[:top_k]
    C_B, C_I = '#e05252', '#4a90d9'

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(
        f'Space Invaders RAM: Bullet Death (n={nb}) vs Enemy Invasion (n={ni})\n'
        f'Bullet: normal DQN play   |   Invasion: lives pinned to 3 via setRAM(0x49,3)',
        fontsize=13, fontweight='bold',
    )
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.38)

    # ── 1. Full 128-byte heatmap ───────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[0, :])
    im = ax_h.imshow(d_score.reshape(8, 16), aspect='auto',
                     cmap='inferno', interpolation='nearest')
    plt.colorbar(im, ax=ax_h, label="Cohen's d", shrink=0.8)
    ax_h.set_title("Cohen's d per RAM byte  (brighter = more discriminative)",
                   fontsize=11)
    ax_h.set_xlabel("address % 16")
    ax_h.set_ylabel("address // 16")
    ax_h.set_xticks(range(16))
    ax_h.set_yticks(range(8))
    ax_h.set_xticklabels([f'{j:X}' for j in range(16)])
    ax_h.set_yticklabels([f'0x{16*i:02X}' for i in range(8)])
    for rank, idx in enumerate(top[:10]):
        r, c = idx // 16, idx % 16
        ax_h.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                                      fill=False, edgecolor='cyan', lw=2))
        ax_h.text(c, r, f'#{rank+1}\n{idx:02X}', ha='center', va='center',
                  fontsize=6, color='cyan', fontweight='bold')

    # ── 2. Bar chart: top-K at terminal frame ─────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, :3])
    x = np.arange(top_k)
    w = 0.38
    ax_bar.bar(x - w/2, mean_b[top], w, yerr=std_b[top],
               label=f'Bullet (n={nb})', color=C_B, capsize=3, alpha=0.85)
    ax_bar.bar(x + w/2, mean_i[top], w, yerr=std_i[top],
               label=f'Invasion (n={ni})', color=C_I, capsize=3, alpha=0.85)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [f'0x{i:02X}\n({i})' + ('\n★' if const[i] else '') for i in top],
        fontsize=7.5,
    )
    ax_bar.set_title(
        f'Top-{top_k} discriminative bytes — mean ± std at terminal frame  '
        '(★ = near-constant in both classes)',
        fontsize=10,
    )
    ax_bar.set_ylabel('RAM value')
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis='y', alpha=0.3)
    for xi, idx in enumerate(top):
        if const[idx]:
            ax_bar.axvspan(xi-.5, xi+.5, color='gold', alpha=0.15, zorder=0)

    # ── 3. Ranking ────────────────────────────────────────────────────────────
    ax_rank = fig.add_subplot(gs[1, 3])
    cols = ['gold' if const[i] else C_B for i in top]
    ax_rank.barh(range(top_k), d_score[top], color=cols, edgecolor='k', lw=0.4)
    ax_rank.set_yticks(range(top_k))
    ax_rank.set_yticklabels([f'0x{i:02X} ({i})' for i in top], fontsize=8)
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("Cohen's d")
    ax_rank.set_title("Discriminative\nranking\n(gold=const)", fontsize=9)
    ax_rank.grid(axis='x', alpha=0.3)

    # ── 4. Temporal trajectories for top-4 bytes ──────────────────────────────
    for plot_i, byte_idx in enumerate(top[:4]):
        row = 2 + plot_i // 2
        col = (plot_i % 2) * 2
        ax  = fig.add_subplot(gs[row, col:col+2])

        for ev in bullet_events:
            h = ev['ram_history'][:, byte_idx].astype(float)
            ax.plot(np.arange(-len(h)+1, 1), h, color=C_B, alpha=0.20, lw=0.8)
        for ev in invasion_events:
            h = ev['ram_history'][:, byte_idx].astype(float)
            ax.plot(np.arange(-len(h)+1, 1), h, color=C_I, alpha=0.20, lw=0.8)

        L = min(
            min(len(e['ram_history']) for e in bullet_events),
            min(len(e['ram_history']) for e in invasion_events),
        )
        t_ax = np.arange(-L+1, 1)

        def _env(events, bidx, length):
            trajs = [ev['ram_history'][-length:, bidx].astype(float)
                     for ev in events if len(ev['ram_history']) >= length]
            if not trajs: return None, None
            arr = np.array(trajs)
            return arr.mean(0), arr.std(0)

        mb, sb = _env(bullet_events,   byte_idx, L)
        mi, si = _env(invasion_events, byte_idx, L)
        if mb is not None:
            ax.plot(t_ax, mb, color=C_B, lw=2.5, label='Bullet mean')
            ax.fill_between(t_ax, mb-sb, mb+sb, color=C_B, alpha=0.15)
        if mi is not None:
            ax.plot(t_ax, mi, color=C_I, lw=2.5, label='Invasion mean')
            ax.fill_between(t_ax, mi-si, mi+si, color=C_I, alpha=0.15)

        ax.axvline(0, color='k', lw=1.5, ls='--', alpha=0.6, label='terminal')
        ax.set_title(
            f"RAM[0x{byte_idx:02X}] ({byte_idx}) — d={d_score[byte_idx]:.2f}"
            + ('  ★' if const[byte_idx] else ''),
            fontsize=10,
        )
        ax.set_xlabel('Steps before terminal  (0 = terminal)')
        ax.set_ylabel('RAM value')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    # ── 5. Sample terminal RGB frames ─────────────────────────────────────────
    ax_b = fig.add_subplot(gs[3, :2])
    ax_i = fig.add_subplot(gs[3, 2:])
    ax_b.imshow(bullet_events[0]['rgb'])
    ax_b.set_title('Bullet death — terminal frame (sample)', fontsize=10)
    ax_b.axis('off')
    ax_i.imshow(invasion_events[0]['rgb'])
    ax_i.set_title('Enemy invasion — terminal frame (sample)', fontsize=10)
    ax_i.axis('off')

    out = 'ram_death_analysis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n[Save] {out}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n_bullet',    type=int,   default=40,
                   help='Number of bullet death episodes to collect')
    p.add_argument('--n_invasion',  type=int,   default=20,
                   help='Number of invasion events to collect (each takes ~6000 raw frames)')
    p.add_argument('--checkpoint',  type=str,   default='pretrained/dqn_cnn.pt')
    p.add_argument('--history_len', type=int,   default=40)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--epsilon',     type=float, default=0.05)
    p.add_argument('--top_k',       type=int,   default=16)
    args = p.parse_args()

    print("=== Collecting BULLET DEATH events (normal DQN play) ===")
    bullet_ev = collect_bullet_events(
        n_episodes  = args.n_bullet,
        checkpoint  = args.checkpoint,
        history_len = args.history_len,
        seed        = args.seed,
        epsilon     = args.epsilon,
    )

    print("\n=== Collecting INVASION events (lives pinned to 3 via setRAM) ===")
    invasion_ev = collect_invasion_events(
        n_events    = args.n_invasion,
        history_len = args.history_len * 4,   # raw frames; bullet history is decision steps
        seed        = args.seed,
    )

    print(f"\nCollected: {len(bullet_ev)} bullet  {len(invasion_ev)} invasion")

    if not bullet_ev or not invasion_ev:
        print("Need both event types. Adjust --n_bullet / --n_invasion.")
        return

    # Use only terminal-frame RAM (last entry of ram_history)
    # Note: bullet history is in decision steps (4 raw frames each),
    #       invasion history is in raw frames — only terminal frame is compared.
    print("\nAnalysing terminal-frame RAM...")
    result = analyse(bullet_ev, invasion_ev, top_k=args.top_k)
    sorted_idx, d_score, mean_b, mean_i, std_b, std_i, const = result

    hdr = (f"{'Rank':>4}  {'Addr':>9}  {'d':>7}  "
           f"{'Mean(B)':>9}  {'Std(B)':>7}  "
           f"{'Mean(I)':>9}  {'Std(I)':>7}  Const?")
    print(f"\nTop-{args.top_k} discriminative RAM bytes:")
    print(hdr)
    print('-' * len(hdr))
    for rank, idx in enumerate(sorted_idx[:args.top_k]):
        print(
            f"{rank+1:>4}  0x{idx:02X} ({idx:3d})  "
            f"{d_score[idx]:>7.2f}  "
            f"{mean_b[idx]:>9.1f}  {std_b[idx]:>7.1f}  "
            f"{mean_i[idx]:>9.1f}  {std_i[idx]:>7.1f}  "
            f"{'★' if const[idx] else ''}"
        )

    print("\nGenerating visualisation...")
    visualize(bullet_ev, invasion_ev, *result, top_k=args.top_k)


if __name__ == '__main__':
    main()
