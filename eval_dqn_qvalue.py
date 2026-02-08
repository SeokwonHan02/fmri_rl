import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from model.dqn import DQN
from dataset import create_train_val_dataloaders

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

    # Agreement validation options
    parser.add_argument('--validate-agreement', action='store_true', default=True,
                        help='Run agreement validation against human data instead of environment evaluation')
    parser.add_argument('--data-dir', type=str,
                        default='/Users/seokwon/research/fMRI_RL/processed_data',
                        help='Base directory containing processed data (for agreement validation)')
    parser.add_argument('--subject', type=str, default='sub_1',
                        choices=['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6'],
                        help='Which subject data to use (for agreement validation)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for data loading (for agreement validation)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers (for agreement validation)')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio (for agreement validation)')

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


def validate_agreement_dqn(model, data_dir, subject, batch_size, num_workers, device, val_split=0.1):
    """
    Validate DQN agreement with human actions on offline dataset

    Args:
        model: DQN model
        data_dir: Directory containing processed data
        subject: Subject ID (e.g., 'sub_1')
        batch_size: Batch size for data loading
        num_workers: Number of data loading workers
        device: Device to run evaluation on
        val_split: Validation split ratio

    Returns:
        Dictionary with agreement statistics
    """
    print("\n" + "="*80)
    print("DQN AGREEMENT VALIDATION")
    print("="*80)
    print(f"Subject: {subject}")
    print(f"Device: {device}")
    print(f"Validation split: {val_split*100:.1f}%")
    print("="*80)

    # Load validation data
    print("\n[1/2] Loading validation data...")
    try:
        train_loader, val_loader = create_train_val_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            subject=subject,
            num_workers=num_workers,
            val_split=val_split
        )
        print(f"✓ Validation set ready ({len(val_loader.dataset)} samples)")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return None

    # Validate
    print("\n[2/2] Running validation...")
    model.eval()

    correct_count = 0
    total_samples = 0
    action_dim = 6
    human_action_counts = np.zeros(action_dim, dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating DQN", ncols=80):
            # Prepare inputs
            state = batch['state'].to(device).float() / 255.0  # Normalize to [0, 1]
            human_action = batch['action'].to(device)

            # Convert one-hot to index if needed
            if human_action.dim() == 2 and human_action.size(1) > 1:
                human_action = human_action.argmax(dim=-1)

            # Collect human action histogram
            for action_idx in human_action.cpu().numpy():
                human_action_counts[action_idx] += 1

            # Get DQN predictions
            q_values = model(state)  # (batch_size, action_dim)
            pred_action = q_values.argmax(dim=-1)  # Greedy action selection

            # Count matches
            correct_count += (pred_action == human_action).sum().item()
            total_samples += state.size(0)

    # Calculate baselines
    most_frequent_action = human_action_counts.argmax()
    most_frequent_count = human_action_counts[most_frequent_action]
    most_frequent_baseline = most_frequent_count / total_samples * 100
    random_baseline = 1 / action_dim * 100

    # Calculate DQN accuracy
    dqn_accuracy = correct_count / total_samples * 100

    # Print action histogram
    print("\n" + "="*80)
    print("HUMAN ACTION DISTRIBUTION")
    print("="*80)
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHT+FIRE', 'LEFT+FIRE']
    for action_idx in range(action_dim):
        count = human_action_counts[action_idx]
        percentage = count / total_samples * 100
        bar = '█' * int(percentage / 2)  # Scale bar to fit screen
        action_name = action_names[action_idx] if action_idx < len(action_names) else f'Action {action_idx}'
        print(f"{action_name:12s} (Action {action_idx}): {count:6,} ({percentage:5.2f}%) {bar}")

    print(f"\nMost Frequent Action: {action_names[most_frequent_action]} (Action {most_frequent_action})")
    print(f"Most Frequent Baseline: {most_frequent_baseline:.2f}%")
    print("="*80)

    # Print results
    print("\n" + "="*80)
    print(f"VALIDATION RESULTS (Total samples: {total_samples:,})")
    print("="*80)
    print(f"\nRandom Chance:           {random_baseline:.2f}%")
    print(f"Most Frequent Baseline:  {most_frequent_baseline:.2f}%")
    print("-"*80)

    beats_baseline = "✓" if dqn_accuracy > most_frequent_baseline else "✗"
    print(f"DQN (pretrained):        {dqn_accuracy:6.2f}%  ({correct_count:,}/{total_samples:,} correct) [{beats_baseline}]")
    print("="*80)

    # Performance vs baseline
    diff = dqn_accuracy - most_frequent_baseline
    print(f"\nPerformance vs. Most Frequent Baseline ({most_frequent_baseline:.2f}%):")
    print("-"*80)
    if diff > 0:
        print(f"DQN (pretrained):    +{diff:.2f}% (BEATS baseline)")
    elif diff < 0:
        print(f"DQN (pretrained):    {diff:.2f}% (below baseline)")
    else:
        print(f"DQN (pretrained):    Same as baseline")
    print("="*80)

    return {
        'accuracy': dqn_accuracy,
        'correct_count': correct_count,
        'total_samples': total_samples,
        'random_baseline': random_baseline,
        'most_frequent_baseline': most_frequent_baseline,
        'human_action_counts': human_action_counts
    }


def main():
    args = get_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Load DQN model
    print("\nLoading DQN model...")
    try:
        model = load_dqn(args.dqn_path, args.action_dim, device)
    except Exception as e:
        print(f"✗ Failed to load DQN: {e}")
        return

    # Branch based on mode
    if args.validate_agreement:
        # Agreement validation mode
        print("="*80)
        print("DQN Agreement Validation Mode")
        print("="*80)
        print(f"Checkpoint: {args.dqn_path}")
        print(f"Data directory: {args.data_dir}")
        print(f"Subject: {args.subject}")
        print(f"Validation split: {args.val_split*100:.1f}%")
        print(f"Device: {device}")
        print("="*80)

        try:
            results = validate_agreement_dqn(
                model=model,
                data_dir=args.data_dir,
                subject=args.subject,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                val_split=args.val_split
            )
        except Exception as e:
            print(f"✗ Agreement validation failed: {e}")
            import traceback
            traceback.print_exc()
            return

    else:
        # Q-value evaluation mode
        print("="*80)
        print("DQN Average Q-Value Evaluation")
        print("="*80)
        print(f"Checkpoint: {args.dqn_path}")
        print(f"Environment: {args.env_name}")
        print(f"Num episodes: {args.num_episodes}")
        print(f"Terminal on life loss: {args.terminal_on_life_loss}")
        print(f"Device: {device}")
        print("="*80)

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
