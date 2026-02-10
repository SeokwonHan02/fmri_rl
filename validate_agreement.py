import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

from dataset import create_train_val_dataloaders
from model import load_pretrained_cnn, BCQ, CQL, BehaviorCloning


def get_validation_args():
    parser = argparse.ArgumentParser(description='Validate model agreement with human actions')

    # Data
    parser.add_argument('--data-dir', type=str,
                        default='/Users/seokwon/research/fMRI_RL/processed_data',
                        help='Base directory containing processed data')
    parser.add_argument('--subject', type=str, default='sub_1',
                        choices=['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6'],
                        help='Which subject data to use')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1 = 10%)')

    # Model paths
    parser.add_argument('--dqn-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/pretrained/dqn_cnn.pt',
                        help='Path to pretrained DQN CNN')
    parser.add_argument('--bc-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/checkpoints/bc/epoch_2.pth',
                        help='Path to trained BC model')
    parser.add_argument('--cql-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/checkpoints/cql/epoch_100.pth',
                        help='Path to trained CQL model')
    parser.add_argument('--bcq-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/checkpoints/bcq/epoch_100.pth',
                        help='Path to trained BCQ model')

    # Model config
    parser.add_argument('--action-dim', type=int, default=6,
                        help='Action dimension (6 for Space Invaders)')

    # Other
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for validation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda, cpu, or auto)')

    # Model selection
    parser.add_argument('--models', type=str, nargs='+', default=['bc', 'cql', 'bcq'],
                        choices=['bc', 'cql', 'bcq'],
                        help='Models to evaluate (space-separated)')

    return parser.parse_args()


def load_model(model_path, model_class, cnn, action_dim, device):
    if not Path(model_path).exists():
        print(f"  ⚠️  Model file not found: {model_path}")
        return None

    try:
        model = model_class(cnn, action_dim=action_dim).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"  ✓ Successfully loaded from {model_path}")
        return model
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return None


def validate_agreement(args):
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("="*80)
    print(f"Validation Agreement Rate Check")
    print("="*80)
    print(f"Subject: {args.subject}")
    print(f"Device: {device}")
    print(f"Validation split: {args.val_split*100:.1f}%")
    print(f"Models to evaluate: {', '.join(args.models)}")
    print("="*80)

    # 1. Load validation data
    print("\n[1/3] Loading validation data...")
    try:
        train_loader, val_loader = create_train_val_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            subject=args.subject,
            num_workers=args.num_workers,
            val_split=args.val_split
        )
        print(f"Validation set ready ({len(val_loader.dataset)} samples)")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 2. Load pretrained CNN
    print("\n[2/3] Loading pretrained CNN...")
    try:
        cnn = load_pretrained_cnn(args.dqn_path).to(device)
        print(f"CNN loaded from {args.dqn_path}")
    except Exception as e:
        print(f"Failed to load CNN: {e}")
        return

    # 3. Load models
    print("\n[3/3] Loading trained models...")
    models = {}

    if 'bc' in args.models:
        print("  Loading BC (Behavior Cloning)...")
        bc_model = load_model(args.bc_path, BehaviorCloning, cnn,
                             args.action_dim, device)
        if bc_model is not None:
            models['bc'] = bc_model

    if 'cql' in args.models:
        print("  Loading CQL (Conservative Q-Learning)...")
        cql_model = load_model(args.cql_path, CQL, cnn,
                              args.action_dim, device)
        if cql_model is not None:
            models['cql'] = cql_model

    if 'bcq' in args.models:
        print("  Loading BCQ (Batch-Constrained Q-Learning)...")
        bcq_model = load_model(args.bcq_path, BCQ, cnn,
                              args.action_dim, device)
        if bcq_model is not None:
            models['bcq'] = bcq_model

    if len(models) == 0:
        print("\nNo models were successfully loaded. Exiting.")
        return

    # 4. Validate
    print("\n" + "="*80)
    print("Running validation...")
    print("="*80)

    # Initialize counters
    correct_counts = {name: 0 for name in models.keys()}
    per_action_correct = {name: np.zeros(args.action_dim, dtype=np.int64) for name in models.keys()}
    total_samples = 0

    # Collect human actions for histogram
    human_action_counts = np.zeros(args.action_dim, dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", ncols=80):
            # Prepare inputs
            state = batch['state'].to(device).float() / 255.0  # Normalize to [0, 1]
            human_action = batch['action'].to(device)

            # Convert one-hot to index if needed
            if human_action.dim() == 2 and human_action.size(1) > 1:
                human_action = human_action.argmax(dim=-1)

            # Collect human action histogram
            for action_idx in human_action.cpu().numpy():
                human_action_counts[action_idx] += 1

            # Get predictions from each model
            for name, model in models.items():
                try:
                    pred = model.get_action(state, deterministic=True)

                    # Convert prediction to tensor if needed
                    if isinstance(pred, (int, np.integer)):
                        pred = torch.tensor([pred], device=device)
                    elif isinstance(pred, (list, np.ndarray)):
                        pred = torch.tensor(pred, device=device)

                    # Ensure pred has same shape as human_action
                    if pred.dim() == 0:
                        pred = pred.unsqueeze(0)

                    # Count total matches
                    matches = (pred == human_action)
                    correct_counts[name] += matches.sum().item()

                    # Count per-action matches
                    for action_idx in range(args.action_dim):
                        action_mask = (human_action == action_idx)
                        per_action_correct[name][action_idx] += (matches & action_mask).sum().item()

                except Exception as e:
                    print(f"\n  ⚠️  Error with {name.upper()}: {e}")
                    continue

            total_samples += state.size(0)

    # 5. Calculate baselines
    most_frequent_action = human_action_counts.argmax()
    most_frequent_count = human_action_counts[most_frequent_action]
    most_frequent_baseline = most_frequent_count / total_samples * 100

    # 6. Print action histogram
    print("\n" + "="*80)
    print("HUMAN ACTION DISTRIBUTION")
    print("="*80)
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHT+FIRE', 'LEFT+FIRE']
    for action_idx in range(args.action_dim):
        count = human_action_counts[action_idx]
        percentage = count / total_samples * 100
        bar = '█' * int(percentage / 2)  # Scale bar to fit screen
        action_name = action_names[action_idx] if action_idx < len(action_names) else f'Action {action_idx}'
        print(f"{action_name:12s} (Action {action_idx}): {count:6,} ({percentage:5.2f}%) {bar}")

    print(f"\nMost Frequent Action: {action_names[most_frequent_action]} (Action {most_frequent_action})")
    print(f"Most Frequent Baseline: {most_frequent_baseline:.2f}% (always predict most frequent)")
    print("="*80)

    # 7. Print overall results
    print("\n" + "="*80)
    print(f"OVERALL VALIDATION RESULTS (Total samples: {total_samples:,})")
    print("="*80)

    # Baselines
    random_baseline = 1/args.action_dim*100
    print(f"\n📊 BASELINES:")
    print(f"  Random Chance:           {random_baseline:.2f}% (uniform random selection)")
    print(f"  Most Frequent Baseline:  {most_frequent_baseline:.2f}% (always predict {action_names[most_frequent_action]})")
    print("-"*80)

    # Sort results by accuracy
    results = []
    for name in models.keys():
        accuracy = correct_counts[name] / total_samples * 100
        results.append((name, accuracy, correct_counts[name]))

    results.sort(key=lambda x: x[1], reverse=True)

    # Display overall results
    model_names = {
        'bc': 'BC  (Habit)',
        'cql': 'CQL (Value)',
        'bcq': 'BCQ (Hybrid)'
    }

    print(f"\n📈 MODEL PERFORMANCE:")
    for name, accuracy, correct in results:
        display_name = model_names.get(name, name.upper())
        beats_baseline = "✓" if accuracy > most_frequent_baseline else "✗"
        vs_random = accuracy - random_baseline
        vs_freq = accuracy - most_frequent_baseline
        print(f"{display_name:15s}: {accuracy:6.2f}%  ({correct:,}/{total_samples:,} correct) [{beats_baseline}]")
        print(f"                    vs Random: {vs_random:+6.2f}%  |  vs Most Freq: {vs_freq:+6.2f}%")

    print("="*80)

    # 8. Per-action accuracy breakdown
    print("\n" + "="*80)
    print("PER-ACTION ACCURACY BREAKDOWN")
    print("="*80)

    for name, accuracy, _ in results:
        display_name = model_names.get(name, name.upper())
        print(f"\n{display_name}:")
        print("-"*80)
        print(f"{'Action':<15s} {'Count':>8s} {'Correct':>8s} {'Accuracy':>10s} {'Bar':>20s}")
        print("-"*80)

        for action_idx in range(args.action_dim):
            action_name = action_names[action_idx] if action_idx < len(action_names) else f'Action {action_idx}'
            count = human_action_counts[action_idx]
            correct = per_action_correct[name][action_idx]

            if count > 0:
                action_acc = correct / count * 100
                bar_length = int(action_acc / 5)  # Scale to 20 chars max
                bar = '█' * bar_length
                print(f"{action_name:<15s} {count:>8,} {correct:>8,} {action_acc:>9.2f}% {bar:>20s}")
            else:
                print(f"{action_name:<15s} {count:>8,} {correct:>8,} {'N/A':>10s}")

        print("-"*80)
        print(f"{'OVERALL':<15s} {total_samples:>8,} {correct_counts[name]:>8,} {accuracy:>9.2f}%")

    print("="*80)


if __name__ == '__main__':
    args = get_validation_args()
    validate_agreement(args)
