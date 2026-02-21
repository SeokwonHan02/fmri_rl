import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm

from dataset import create_train_val_dataloaders
from model import load_pretrained_cnn, BehaviorCloning

def get_args():
    parser = argparse.ArgumentParser(description='Analyze BC and BCQ performance on validation data')

    # Data
    parser.add_argument('--data-dir', type=str,
                        default='/Users/seokwon/research/fMRI_RL/processed_data_frameskip_4',
                        help='Base directory containing processed data')
    parser.add_argument('--subject', type=str, default='sub_1',
                        choices=['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6'],
                        help='Which subject data to use')
    parser.add_argument('--val-file-idx', type=int, default=10,
                        help='Index of file to use for validation (0-10 for 11 files)')

    # Model paths
    parser.add_argument('--dqn-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/pretrained/dqn_cnn.pt',
                        help='Path to pretrained DQN checkpoint')
    parser.add_argument('--bc-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/pretrained/bc_653_15.pth',
                        help='Path to trained BC model')

    # Other
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')

    return parser.parse_args()


def analyze_bc_ranking(bc, val_loader, device):
    """
    Analyze BC model: compute ranking of human-selected actions

    Returns:
        rank_distribution: array of shape (6,) counting how many times human action was rank 1, 2, ..., 6
    """
    bc.eval()
    rank_counts = np.zeros(6, dtype=np.int64)  # rank 1, 2, 3, 4, 5, 6
    total_samples = 0

    print("\nAnalyzing BC action rankings...")
    for batch in tqdm(val_loader, desc="Processing batches", ncols=80):
        state = batch['state'].to(device).float() / 255.0
        action = batch['action'].to(device)

        # Convert action to index if one-hot
        if action.dim() == 2 and action.size(1) > 1:
            action_idx = action.argmax(dim=-1)
        else:
            action_idx = action.long()

        with torch.no_grad():
            # Get BC logits
            bc_logits = bc(state)  # (batch, 6)

            # For each sample, compute rank of human action
            batch_size = bc_logits.size(0)
            for i in range(batch_size):
                logits = bc_logits[i]  # (6,)
                human_action = action_idx[i].item()

                # Compute rank: how many actions have higher logits than human action?
                human_logit = logits[human_action]
                rank = (logits > human_logit).sum().item() + 1  # +1 because rank starts from 1

                rank_counts[rank - 1] += 1  # -1 for 0-indexing
                total_samples += 1

    return rank_counts, total_samples


def analyze_bcq_masking(bc, val_loader, device, thresholds):
    """
    Analyze BCQ masking: compute how many human actions are masked for different thresholds

    Args:
        bc: BC model (used for imitation network)
        val_loader: validation data loader
        device: device
        thresholds: list of threshold values to test

    Returns:
        masking_stats: dict mapping threshold -> (num_masked, total_samples, percentage, mean_allowed, std_allowed)
    """
    bc.eval()
    masking_stats = {}

    # Collect all data first
    all_states = []
    all_actions = []

    print("\nCollecting validation data...")
    for batch in tqdm(val_loader, desc="Loading batches", ncols=80):
        state = batch['state'].to(device).float() / 255.0
        action = batch['action'].to(device)

        # Convert action to index if one-hot
        if action.dim() == 2 and action.size(1) > 1:
            action_idx = action.argmax(dim=-1)
        else:
            action_idx = action.long()

        all_states.append(state)
        all_actions.append(action_idx)

    # Concatenate all batches
    all_states = torch.cat(all_states, dim=0)
    all_actions = torch.cat(all_actions, dim=0)
    total_samples = all_actions.size(0)

    # Compute imitation logits once
    print("Computing BC imitation logits...")
    with torch.no_grad():
        imitation_logits = bc(all_states)  # (total_samples, 6)
        imitation_probs = F.softmax(imitation_logits, dim=-1)  # (total_samples, 6)

    # Analyze masking for each threshold
    print("Analyzing BCQ masking for different thresholds...")
    for threshold in tqdm(thresholds, desc="Thresholds", ncols=80):
        # Compute mask
        max_prob = imitation_probs.max(dim=-1, keepdim=True)[0]  # (total_samples, 1)
        mask = imitation_probs > (max_prob * threshold)  # (total_samples, 6)

        # Count how many human actions are masked
        num_masked = 0
        for i in range(total_samples):
            human_action = all_actions[i].item()
            is_allowed = mask[i, human_action].item()
            if not is_allowed:
                num_masked += 1

        # Count average number of allowed actions per state
        num_allowed_per_state = mask.sum(dim=-1).float()  # (total_samples,)
        mean_allowed_actions = num_allowed_per_state.mean().item()
        std_allowed_actions = num_allowed_per_state.std().item()

        percentage = (num_masked / total_samples * 100) if total_samples > 0 else 0
        masking_stats[threshold] = (num_masked, total_samples, percentage, mean_allowed_actions, std_allowed_actions)

    return masking_stats


def main():
    args = get_args()
    device = torch.device(args.device)

    print("="*80)
    print("BC & BCQ PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"Subject: {args.subject}")
    print(f"Validation file index: {args.val_file_idx}")
    print(f"Device: {device}")
    print("="*80)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader = create_train_val_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        subject=args.subject,
        num_workers=args.num_workers,
        val_file_idx=args.val_file_idx
    )
    print(f"✓ Validation batches: {len(val_loader)}")

    # Load BC model
    print("\nLoading BC model...")
    dummy_cnn = load_pretrained_cnn(args.dqn_path, freeze=True)
    bc = BehaviorCloning(dummy_cnn, action_dim=6)
    bc.load_state_dict(torch.load(args.bc_path, map_location=device))
    bc = bc.to(device)
    bc.eval()
    print("✓ BC loaded")

    # ===== BC Analysis: Action Ranking =====
    print("\n" + "="*80)
    print("PART 1: BC ACTION RANKING ANALYSIS")
    print("="*80)

    rank_counts, total_samples = analyze_bc_ranking(bc, val_loader, device)

    print("\n" + "="*80)
    print("BC ACTION RANKING RESULTS")
    print("="*80)
    print(f"Total samples: {total_samples:,}\n")
    print("Ranking Distribution (How BC ranks human-selected actions):")
    print("-"*80)

    for rank in range(1, 7):
        count = rank_counts[rank - 1]
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        bar_length = int(percentage / 2)  # Scale to 50 chars max
        bar = '█' * bar_length

        # Emoji for rank 1
        emoji = "🥇" if rank == 1 else "  "
        print(f"{emoji} Rank {rank}: {count:>8,} ({percentage:>6.2f}%) {bar}")

    print("-"*80)
    print(f"   TOTAL:  {total_samples:>8,} (100.00%)")

    # Summary statistics
    top1_percentage = (rank_counts[0] / total_samples * 100) if total_samples > 0 else 0
    top3_count = rank_counts[0] + rank_counts[1] + rank_counts[2]
    top3_percentage = (top3_count / total_samples * 100) if total_samples > 0 else 0

    print("\nSummary:")
    print(f"  Top-1 Accuracy (human action ranked #1 by BC): {top1_percentage:.2f}%")
    print(f"  Top-3 Accuracy (human action in top 3):       {top3_percentage:.2f}%")

    # ===== BCQ Analysis: Masking with Different Thresholds =====
    print("\n" + "="*80)
    print("PART 2: BCQ MASKING ANALYSIS (Different Thresholds)")
    print("="*80)

    thresholds = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    masking_stats = analyze_bcq_masking(bc, val_loader, device, thresholds)

    print("\n" + "="*80)
    print("BCQ MASKING RESULTS")
    print("="*80)
    print(f"Total samples: {total_samples:,}\n")
    print("Human Actions Masked by BCQ (using BC as imitation network):")
    print("-"*100)
    print(f"{'Threshold':<12s} {'Masked':<12s} {'Allowed':<12s} {'Masked %':<12s} {'Allowed %':<12s} {'Avg Allowed Actions':<20s}")
    print("-"*100)

    for threshold in thresholds:
        num_masked, total, percentage_masked, mean_allowed, std_allowed = masking_stats[threshold]
        num_allowed = total - num_masked
        percentage_allowed = 100 - percentage_masked

        print(f"{threshold:<12.2f} {num_masked:<12,} {num_allowed:<12,} {percentage_masked:<12.2f} {percentage_allowed:<12.2f} {mean_allowed:<6.2f} ± {std_allowed:<6.2f}")

    print("-"*100)

    # Visualization
    print("\nVisualization (Masking Percentage):")
    print("-"*80)
    for threshold in thresholds:
        num_masked, total, percentage_masked, mean_allowed, std_allowed = masking_stats[threshold]
        bar_length = int(percentage_masked / 2)  # Scale to 50 chars max
        bar = '█' * bar_length
        print(f"Threshold {threshold:.2f}: {percentage_masked:>6.2f}% {bar}")

    print("\nVisualization (Average Allowed Actions per State):")
    print("-"*80)
    for threshold in thresholds:
        num_masked, total, percentage_masked, mean_allowed, std_allowed = masking_stats[threshold]
        bar_length = int(mean_allowed * 8)  # Scale: 6 actions = 48 chars
        bar = '█' * bar_length
        print(f"Threshold {threshold:.2f}: {mean_allowed:>4.2f} / 6 actions {bar}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
