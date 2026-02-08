import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from model.dqn import DQN

# Import and register ALE
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Warning: ale_py not found. Make sure to install it: pip install ale-py")
except Exception as e:
    print(f"Warning: Failed to register ALE environments: {e}")


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate DQN average Q-value')

    parser.add_argument('--dqn-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/pretrained/dqn_cnn.pt',
                        help='Path to pretrained DQN checkpoint')
    parser.add_argument('--env-name', type=str,
                        default='ALE/SpaceInvaders-v5',
                        help='Environment name')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--action-dim', type=int, default=6,
                        help='Action dimension')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for environment')
    parser.add_argument('--terminal-on-life-loss', action='store_true',
                        help='Terminate episode on life loss (standard for training)')

    return parser.parse_args()


def load_dqn(checkpoint_path, action_dim, device):
    """Load DQN model from checkpoint"""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create model
    model = DQN(action_dim=action_dim).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract policy_net if checkpoint has the full training state
    if 'policy_net' in checkpoint:
        state_dict = checkpoint['policy_net']
        print(f"✓ Loaded from training checkpoint (epoch/step info available)")
        if 'frame_count' in checkpoint:
            print(f"  Frame count: {checkpoint['frame_count']:,}")
        if 'train_count' in checkpoint:
            print(f"  Train count: {checkpoint['train_count']:,}")
    else:
        state_dict = checkpoint

    # Load state dict
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"✓ Loaded DQN from {checkpoint_path}")
    return model


def make_atari_env(env_name='ALE/SpaceInvaders-v5', seed=0, terminal_on_life_loss=False):
    """Create Atari environment with standard preprocessing"""
    env = gym.make(env_name, frameskip=1, render_mode='rgb_array')  # frameskip=1 추가!
    
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=1,
        screen_size=84,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=True,
        scale_obs=False
    )
    
    env = FrameStackObservation(env, stack_size=4)
    env.reset(seed=seed)
    
    return env


def evaluate_dqn_qvalue(model, env_name, device, num_episodes=10, seed=0, terminal_on_life_loss=False):
    """
    Evaluate DQN average Q-value over episodes

    Args:
        model: DQN model
        env_name: Name of the environment
        device: Device to run evaluation on
        num_episodes: Number of episodes to run
        seed: Random seed for environment
        terminal_on_life_loss: Terminate episode on life loss

    Returns:
        Dictionary with Q-value statistics and episode rewards
    """
    # Create environment
    env = make_atari_env(env_name, seed, terminal_on_life_loss)

    episode_rewards = []
    episode_lengths = []
    all_q_values = []

    model.eval()

    print(f"\nEvaluating DQN for {num_episodes} episodes...")
    print("="*80)

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_q_values = []
        done = False

        pbar = tqdm(desc=f"Episode {episode+1}/{num_episodes}", leave=False)

        while not done:
            # Convert state to tensor
            # State from env is (4, 84, 84) uint8
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device).float() / 255.0

            # Get Q-values and action from model (epsilon-greedy with epsilon=0.05, same as DQN_variants eval)
            with torch.no_grad():
                q_values = model(state_tensor)  # (1, action_dim)

                # Epsilon-greedy action selection (epsilon=0.05)
                if np.random.random() < 0.05:
                    action = env.action_space.sample()
                else:
                    action = q_values.argmax(dim=-1).item()

                # Store Q-values
                episode_q_values.append(q_values.cpu().numpy())

            # Step environment
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            pbar.update(1)

        pbar.close()

        # Store episode stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        all_q_values.extend(episode_q_values)

        # Print episode summary
        avg_q_ep = np.mean([q.max() for q in episode_q_values])  # Avg max Q
        print(f"Episode {episode+1:2d}: Reward={episode_reward:7.1f}, "
              f"Length={episode_length:4d}, Avg Max Q={avg_q_ep:6.2f}")

    env.close()

    # Calculate overall statistics
    all_q_values = np.concatenate(all_q_values, axis=0)  # (total_steps, action_dim)

    results = {
        # Reward stats
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),

        # Q-value stats
        'mean_q_value': np.mean(all_q_values),  # Average over all Q-values
        'mean_max_q': np.mean(all_q_values.max(axis=1)),  # Average max Q per step
        'std_q_value': np.std(all_q_values),
        'min_q_value': np.min(all_q_values),
        'max_q_value': np.max(all_q_values),
    }

    return results


def main():
    args = get_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("="*80)
    print("DQN Average Q-Value Evaluation")
    print("="*80)
    print(f"Checkpoint: {args.dqn_path}")
    print(f"Environment: {args.env_name}")
    print(f"Num episodes: {args.num_episodes}")
    print(f"Terminal on life loss: {args.terminal_on_life_loss}")
    print(f"Device: {device}")
    print("="*80)

    # Load DQN model
    print("\nLoading DQN model...")
    try:
        model = load_dqn(args.dqn_path, args.action_dim, device)
    except Exception as e:
        print(f"✗ Failed to load DQN: {e}")
        return

    # Evaluate Q-values
    try:
        results = evaluate_dqn_qvalue(
            model=model,
            env_name=args.env_name,
            device=device,
            num_episodes=args.num_episodes,
            seed=args.seed,
            terminal_on_life_loss=args.terminal_on_life_loss
        )
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nReward Statistics ({args.num_episodes} episodes):")
    print(f"  Mean:   {results['mean_reward']:8.2f} ± {results['std_reward']:.2f}")
    print(f"  Min:    {results['min_reward']:8.2f}")
    print(f"  Max:    {results['max_reward']:8.2f}")
    print(f"  Length: {results['mean_length']:8.1f} steps/episode")

    print(f"\nQ-Value Statistics:")
    print(f"  Mean Q (all):  {results['mean_q_value']:8.2f} ± {results['std_q_value']:.2f}")
    print(f"  Mean Max Q:    {results['mean_max_q']:8.2f}")
    print(f"  Min Q:         {results['min_q_value']:8.2f}")
    print(f"  Max Q:         {results['max_q_value']:8.2f}")
    print("="*80)


if __name__ == '__main__':
    main()
