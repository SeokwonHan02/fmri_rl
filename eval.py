import torch
import numpy as np
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


def evaluate_agent(model, env_name, device, num_episodes=10, seed=0, deterministic=True):
    """
    Evaluate an agent in the real environment

    Args:
        model: The model to evaluate (BC, BCQ, or CQL)
        env_name: Name of the environment
        device: Device to run evaluation on
        num_episodes: Number of episodes to run
        seed: Random seed for environment
        deterministic: Whether to use deterministic action selection

    Returns:
        Dictionary with mean, std, min, max rewards
    """
    # Create environment
    env = make_atari_env(env_name, seed)

    episode_rewards = []
    episode_lengths = []

    model.eval()

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Convert state to tensor and add batch dimension
            # State from env is (4, 84, 84) uint8
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device).float() / 255.0

            # Get action from model
            with torch.no_grad():
                action = model.get_action(state_tensor.squeeze(0), deterministic=deterministic)

            # Step environment
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
    }
