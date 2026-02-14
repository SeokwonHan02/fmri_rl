import torch
import numpy as np
import random
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

# Import and register ALE
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Warning: ale_py not found. Make sure to install it: pip install ale-py")
except Exception as e:
    print(f"Warning: Failed to register ALE environments: {e}")


def make_atari_env(env_name='SpaceInvadersNoFrameskip-v4', seed=0):
    """Create Atari environment with standard preprocessing"""
    # Create environment (rgb_array for headless server environments)
    env = gym.make(env_name, render_mode='rgb_array')

    # Atari preprocessing: grayscale, resize, etc.
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,  # Apply frame skip of 4
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False
    )

    # Stack 4 frames
    env = FrameStackObservation(env, stack_size=4)

    # Set seed
    env.reset(seed=seed)

    return env


def evaluate_agent(model, env_name, device, num_episodes=10, seed=None, deterministic=True, track_masking=False):
    """
    Evaluate an agent in the real environment

    Args:
        model: The model to evaluate (BC, BCQ, or CQL)
        env_name: Name of the environment
        device: Device to run evaluation on
        num_episodes: Number of episodes to run
        seed: Random seed for environment (default: random)
        deterministic: Whether to use deterministic action selection
        track_masking: Whether to track BCQ action masking statistics

    Returns:
        Dictionary with mean, std, min, max rewards (and masking stats if track_masking=True)
    """
    # Generate random seed if not specified
    if seed is None:
        seed = random.randint(0, 999999)

    # Create environment
    env = make_atari_env(env_name, seed)

    episode_rewards = []
    episode_lengths = []

    # BCQ masking statistics
    if track_masking:
        num_allowed_actions_list = []

    model.eval()

    for episode in range(num_episodes):
        # Generate unique seed for each episode
        episode_seed = seed + episode
        state, info = env.reset(seed=episode_seed)
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Convert state to tensor and add batch dimension
            # State from env is (4, 84, 84) uint8
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device).float() / 255.0

            # Get action from model
            with torch.no_grad():
                # If BCQ and tracking masking, compute masking statistics
                if track_masking and hasattr(model, 'imitation_network'):
                    # BCQ model - manually compute action with masking stats
                    q_values, imitation_logits = model.forward(state_tensor)

                    # Get imitation probabilities
                    imitation_probs = torch.softmax(imitation_logits, dim=-1)

                    # Compute mask
                    max_prob = imitation_probs.max(dim=-1, keepdim=True)[0]
                    mask = imitation_probs > (max_prob * model.threshold)

                    # Count allowed actions
                    num_allowed = mask.sum().item()
                    num_allowed_actions_list.append(num_allowed)

                    # Get action (same as BCQ.get_action)
                    masked_q = q_values.clone()
                    masked_q[~mask] = -float('inf')
                    action = masked_q.argmax(dim=-1).item()
                else:
                    # Standard action selection
                    action = model.get_action(state_tensor.squeeze(0), deterministic=deterministic)

            # Step environment
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()

    result = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
    }

    # Add masking statistics if tracked
    if track_masking and len(num_allowed_actions_list) > 0:
        result['mean_allowed_actions'] = np.mean(num_allowed_actions_list)
        result['std_allowed_actions'] = np.std(num_allowed_actions_list)

    return result
