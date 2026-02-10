import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Offline RL for Space Invaders')

    # Algorithm selection
    parser.add_argument('--algo', type=str, default='bc', choices=['bc', 'bcq', 'cql'],
                        help='Algorithm to use: bc (Behavior Cloning), bcq (BCQ), or cql (CQL)')

    # Data
    parser.add_argument('--data-dir', type=str,
                        default='/Users/seokwon/research/fMRI_RL/processed_data',
                        help='Base directory containing processed data')
    parser.add_argument('--subject', type=str, default='sub_1',
                        choices=['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6'],
                        help='Which subject data to use')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Model
    parser.add_argument('--dqn-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/pretrained/dqn_cnn.pt',
                        help='Path to pretrained DQN parameters')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr-decay-factor', type=float, default=0.1,
                        help='Final learning rate factor (1.0 = no decay)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for RL')

    # CQL specific
    parser.add_argument('--cql-alpha', type=float, default=1.0,
                        help='CQL regularization weight')
    parser.add_argument('--target-update-freq', type=int, default=1000,
                        help='Target network update frequency')

    # BCQ specific
    parser.add_argument('--bcq-threshold', type=float, default=0.3,
                        help='BCQ action filtering threshold')
    parser.add_argument('--bc-path', type=str, default='',
                        help='Path to pretrained BC model (if provided, BC network will be frozen)')

    # BC and BCQ behavior cloning
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing for BC loss (0.0 = no smoothing)')
    parser.add_argument('--logit-div', type=float, default=2.0,
                        help='Logit division for temperature scaling')
    parser.add_argument('--class-weight-exponent', type=float, default=0.5,
                        help='Exponent for class weights (0.0 = no weight, 0.5 = sqrt, 1.0 = inverse frequency)')

    # Logging and saving
    parser.add_argument('--save-dir', type=str, default='/Users/seokwon/research/fMRI_RL/checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging interval (steps)')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Model saving interval (epochs)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    # Evaluation
    parser.add_argument('--env-name', type=str, default='SpaceInvadersNoFrameskip-v4',
                        help='Environment name for evaluation')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes to run for evaluation')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='Evaluation interval (epochs)')

    args = parser.parse_args()
    return args
