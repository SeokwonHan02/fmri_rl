import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Offline RL for Space Invaders')

    # Algorithm selection
    parser.add_argument('--algo', type=str, default='bc',
                        choices=['bc', 'bcq', 'cql', 'prob_cql', 'ensemble_dqn'],
                        help='Algorithm to use: bc, bcq, cql, prob_cql, or ensemble_dqn')

    # Data
    parser.add_argument('--data-dir', type=str,
                        default='/Users/seokwon/research/fMRI_RL/processed_data_frameskip_4',
                        help='Base directory containing processed data')
    parser.add_argument('--subject', type=str, default='sub_1',
                        choices=['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6'],
                        help='Which subject data to use')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--val-file-idx', type=int, default=9,
                        help='Index of validation file (0-based)')
    parser.add_argument('--test-file-idx', type=int, default=10,
                        help='Index of test file (0-based)')

    # Model
    parser.add_argument('--dqn-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/pretrained/dqn_cnn.pt',
                        help='Path to pretrained DQN parameters (encoder always frozen)')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for RL')
    parser.add_argument('--reward-scale', type=float, default=0.1,
                        help='Reward scaling factor')

    # CQL / ProbCQL shared
    parser.add_argument('--cql-alpha', type=float, default=0.3,
                        help='CQL regularization weight (shared by cql and prob_cql)')

    # ProbCQL specific
    parser.add_argument('--prob-cql-beta-kl', type=float, default=1e-3,
                        help='KL divergence (Information Bottleneck) weight for prob_cql')
    parser.add_argument('--target-update-freq', type=int, default=100,
                        help='Target network update frequency')

    # EnsembleDQN specific
    parser.add_argument('--num-heads', type=int, default=50,
                        help='Number of MLP heads in EnsembleDQN (default: 50)')
    parser.add_argument('--mask-prob', type=float, default=1.0,
                        help='Bernoulli bootstrap mask probability per (sample, head) pair '
                             '(1.0 = all samples per head, 0.5 = half the batch per head)')
    parser.add_argument('--ce-lambda', type=float, default=1.0,
                        help='Weight for CE regularization loss in EnsembleDQN (total = td + λ * ce)')

    # BCQ specific
    parser.add_argument('--bcq-threshold', type=float, default=0.3,
                        help='BCQ action filtering threshold')
    parser.add_argument('--bc-path', type=str, default='',
                        help='Path to pretrained BC model for BCQ initialization (BC network always frozen)')

    # BC and BCQ behavior cloning
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing for BC loss (0.0 = no smoothing)')
    parser.add_argument('--class-weight-exponent', type=float, default=0.5,
                        help='Exponent for class weights (0.0 = no weight, 0.5 = sqrt, 1.0 = inverse frequency)')
    parser.add_argument('--fire-loss-weight', type=float, default=1.5,
                        help='Weight for fire loss in BC (to balance with move loss). '
                             'Default 1.5 since fire has fewer classes than move.')
    parser.add_argument('--move-loss-weight', type=float, default=1.0,
                        help='Weight for move loss in BC.')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic action selection during evaluation (default: True)')
    parser.add_argument('--stochastic', dest='deterministic', action='store_false',
                        help='Use stochastic action selection during evaluation')

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
