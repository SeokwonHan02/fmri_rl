import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from dataset import create_train_val_dataloaders
from model import load_pretrained_cnn, BehaviorCloning, BCQ, CQL
from model.dqn import DQN


def get_args():
    parser = argparse.ArgumentParser(description='Compute cognitive indices from model log probabilities')

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
                        default='/Users/seokwon/research/fMRI_RL/pretrained/bc.pth',
                        help='Path to trained BC model')
    parser.add_argument('--cql-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/pretrained/cql.pth',
                        help='Path to trained CQL model')
    parser.add_argument('--bcq-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/pretrained/bcq.pth',
                        help='Path to trained BCQ model')

    # Other
    parser.add_argument('--bcq-threshold', type=float, default = 0.04,
                        help='Threshold for action selection')
    parser.add_argument('--bcq-masked-z-score', type=float, default=-3.0,
                        help='Z-score value assigned to masked BCQ actions')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')

    return parser.parse_args()


def load_models(args, device):
    """Load all models (DQN, BC, CQL, BCQ)"""
    print("Loading models...")

    # Load DQN from policy_net in dqn_cnn.pt
    print(f"  Loading DQN from {args.dqn_path}")
    dqn = DQN(action_dim=6)
    checkpoint = torch.load(args.dqn_path, map_location=device)
    # Extract policy_net state_dict
    if 'policy_net' in checkpoint:
        dqn.load_state_dict(checkpoint['policy_net'])
    else:
        dqn.load_state_dict(checkpoint)
    dqn = dqn.to(device)
    dqn.eval()

    # Load BC (checkpoint already contains CNN params)
    print(f"  Loading BC from {args.bc_path}")
    # Need to create a dummy CNN for BC initialization (will be overwritten by checkpoint)
    dummy_cnn = load_pretrained_cnn(args.dqn_path, freeze=True)
    bc = BehaviorCloning(dummy_cnn, action_dim=6)
    bc.load_state_dict(torch.load(args.bc_path, map_location=device))
    bc = bc.to(device)
    bc.eval()
    print()

    # Load CQL (checkpoint already contains CNN params)
    print(f"  Loading CQL from {args.cql_path}")
    dummy_cnn = load_pretrained_cnn(args.dqn_path, freeze=True)
    cql = CQL(dummy_cnn, action_dim=6, alpha=0.2)  # alpha not used for inference
    cql.load_state_dict(torch.load(args.cql_path, map_location=device))
    cql = cql.to(device)
    cql.eval()
    print()

    # Load BCQ (checkpoint already contains CNN and imitation_network params)
    print(f"  Loading BCQ from {args.bcq_path}")
    dummy_cnn = load_pretrained_cnn(args.dqn_path, freeze=True)
    # Note: bc_path not needed - BCQ checkpoint already contains imitation_network
    bcq = BCQ(dummy_cnn, action_dim=6, threshold=args.bcq_threshold)
    bcq.load_state_dict(torch.load(args.bcq_path, map_location=device))
    bcq = bcq.to(device)
    bcq.eval()
    print()

    print("✓ All models loaded successfully\n")
    return dqn, bc, cql, bcq


def compute_z_scores(state, action, dqn, bc, cql, bcq, masked_z_value=-3.0):
    """
    Compute Z-scores for human-selected actions across all models

    Z-score normalizes Q-values/logits within each state, making models comparable
    regardless of their absolute scale.

    Z(s, a) = (Q(s, a) - mean(Q(s))) / std(Q(s))

    Args:
        state: (batch, 4, 84, 84) tensor
        action: (batch,) or (batch, 6) tensor
        dqn, bc, cql, bcq: model instances
        masked_z_value: Z-score value assigned to masked BCQ actions (default: -3.0)

    Returns:
        dict with z_scores for each model
    """
    # Convert action to index if one-hot
    if action.dim() == 2 and action.size(1) > 1:
        action_idx = action.argmax(dim=-1)
    else:
        action_idx = action.long()

    with torch.no_grad():
        # Small epsilon to prevent division by zero
        eps = 1e-8

        # ===== DQN: Q-values -> Z-score =====
        dqn_q = dqn(state)  # (batch, 6)
        dqn_mean = dqn_q.mean(dim=-1, keepdim=True)  # (batch, 1)
        dqn_std = dqn_q.std(dim=-1, keepdim=True) + eps  # (batch, 1)
        dqn_z = (dqn_q - dqn_mean) / dqn_std  # (batch, 6)
        dqn_z_human = dqn_z.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # (batch,)

        # ===== BC: action logits -> Z-score =====
        bc_logits = bc(state)  # (batch, 6)
        bc_mean = bc_logits.mean(dim=-1, keepdim=True)  # (batch, 1)
        bc_std = bc_logits.std(dim=-1, keepdim=True) + eps  # (batch, 1)
        bc_z = (bc_logits - bc_mean) / bc_std  # (batch, 6)
        bc_z_human = bc_z.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # (batch,)

        # ===== CQL: Q-values -> Z-score =====
        cql_q = cql(state)  # (batch, 6)
        cql_mean = cql_q.mean(dim=-1, keepdim=True)  # (batch, 1)
        cql_std = cql_q.std(dim=-1, keepdim=True) + eps  # (batch, 1)
        cql_z = (cql_q - cql_mean) / cql_std  # (batch, 6)
        cql_z_human = cql_z.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # (batch,)

        # ===== BCQ: Q-values with action masking -> Z-score =====
        bcq_q, bcq_imitation_logits = bcq(state)  # (batch, 6), (batch, 6)

        # Compute imitation probabilities for masking
        imitation_probs = F.softmax(bcq_imitation_logits, dim=-1)

        # Apply BCQ masking threshold
        max_prob = imitation_probs.max(dim=-1, keepdim=True)[0]
        mask = imitation_probs > (max_prob * bcq.threshold)  # (batch, 6)

        # Compute Z-score ONLY on unmasked Q-values
        mask_float = mask.float()  # (batch, 6)
        mask_count = mask_float.sum(dim=-1, keepdim=True) + eps  # (batch, 1)

        # Mean of unmasked Q-values
        masked_q_sum = (bcq_q * mask_float).sum(dim=-1, keepdim=True)  # (batch, 1)
        bcq_mean = masked_q_sum / mask_count  # (batch, 1)

        # Std of unmasked Q-values
        masked_q_centered = (bcq_q - bcq_mean) * mask_float  # (batch, 6)
        masked_q_var = (masked_q_centered ** 2).sum(dim=-1, keepdim=True) / mask_count  # (batch, 1)
        bcq_std = torch.sqrt(masked_q_var) + eps  # (batch, 1)

        bcq_z = (bcq_q - bcq_mean) / bcq_std  # (batch, 6)
        bcq_z = torch.where(mask, bcq_z, torch.full_like(bcq_z, masked_z_value))

        bcq_z_human = bcq_z.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # (batch,)

    # Return Z-scores for human-selected actions
    return {
        'dqn_z': dqn_z_human,
        'bc_z': bc_z_human,
        'cql_z': cql_z_human,
        'bcq_z': bcq_z_human,
    }


def main():
    args = get_args()
    device = torch.device(args.device)

    print("="*80)
    print("COGNITIVE INDICES COMPUTATION (Z-Score Based)")
    print("="*80)
    print(f"Subject: {args.subject}")
    print(f"Validation file index: {args.val_file_idx}")
    print(f"BCQ masked Z-score value: {args.bcq_masked_z_score}")
    print(f"Device: {device}")
    print("="*80 + "\n")

    # Load data
    print("Loading data...")
    train_loader, val_loader = create_train_val_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        subject=args.subject,
        num_workers=args.num_workers,
        val_file_idx=args.val_file_idx
    )
    print(f"✓ Validation batches: {len(val_loader)}\n")

    # Load models
    dqn, bc, cql, bcq = load_models(args, device)

    # Compute Z-scores on validation data
    print("Computing Z-scores on validation data...")
    all_dqn_z = []
    all_bc_z = []
    all_cql_z = []
    all_bcq_z = []

    for batch in tqdm(val_loader, desc="Processing batches", ncols=80):
        state = batch['state'].to(device).float() / 255.0
        action = batch['action'].to(device)

        results = compute_z_scores(state, action, dqn, bc, cql, bcq,
                                   masked_z_value=args.bcq_masked_z_score)

        all_dqn_z.append(results['dqn_z'].cpu())
        all_bc_z.append(results['bc_z'].cpu())
        all_cql_z.append(results['cql_z'].cpu())
        all_bcq_z.append(results['bcq_z'].cpu())

    # Concatenate all batches
    dqn_z = torch.cat(all_dqn_z).numpy()
    bc_z = torch.cat(all_bc_z).numpy()
    cql_z = torch.cat(all_cql_z).numpy()
    bcq_z = torch.cat(all_bcq_z).numpy()

    # Filter out masked actions from BCQ Z-scores (masked actions have Z-score = args.bcq_masked_z_score)
    # Use a small epsilon to avoid floating point comparison issues
    masked_threshold = args.bcq_masked_z_score + 0.1
    bcq_z_unmasked = bcq_z[bcq_z > masked_threshold]

    print(f"✓ Computed Z-scores for {len(dqn_z):,} samples\n")

    # Compute cognitive indices based on Z-scores
    print("="*80)
    print("COMPUTING COGNITIVE INDICES (Z-Score Based)")
    print("="*80)

    I_habit = bc_z - dqn_z        # BC - DQN
    I_pessimism = cql_z - dqn_z   # CQL - DQN
    I_prudence = bcq_z - dqn_z    # BCQ - DQN

    # Print Z-score statistics
    print("\nZ-Score Statistics (human-selected actions):")
    print(f"  Z_DQN:  mean={np.mean(dqn_z):.4f}, std={np.std(dqn_z):.4f}")
    print(f"  Z_BC:   mean={np.mean(bc_z):.4f}, std={np.std(bc_z):.4f}")
    print(f"  Z_CQL:  mean={np.mean(cql_z):.4f}, std={np.std(cql_z):.4f}")
    print(f"  Z_BCQ:  mean={np.mean(bcq_z):.4f}, std={np.std(bcq_z):.4f} (all samples)")
    print(f"          mean={np.mean(bcq_z_unmasked):.4f}, std={np.std(bcq_z_unmasked):.4f}, min={np.min(bcq_z_unmasked):.4f}, max={np.max(bcq_z_unmasked):.4f} (unmasked only, {len(bcq_z_unmasked)}/{len(bcq_z)} samples)")

    # Z-score correlations
    print("\n" + "="*80)
    print("Z-SCORE CORRELATIONS (Cross-Model Analysis)")
    print("="*80)

    # Create correlation matrix (using all samples including masked for BCQ)
    z_scores_all = np.vstack([dqn_z, bc_z, cql_z, bcq_z])
    z_corr_all = np.corrcoef(z_scores_all)

    z_names = ['Z_DQN', 'Z_BC', 'Z_CQL', 'Z_BCQ']
    print("\nCorrelation Matrix - All Samples (Pearson):")
    print(f"{'':>12s}", end='')
    for name in z_names:
        print(f"{name:>12s}", end='')
    print()
    print("-"*60)

    for i, name in enumerate(z_names):
        print(f"{name:>12s}", end='')
        for j in range(len(z_names)):
            print(f"{z_corr_all[i, j]:>12.4f}", end='')
        print()

    # Also compute correlations excluding masked BCQ actions
    bcq_unmasked_mask = bcq_z > masked_threshold
    bcq_masked_mask = bcq_z <= masked_threshold

    dqn_z_filtered = dqn_z[bcq_unmasked_mask]
    bc_z_filtered = bc_z[bcq_unmasked_mask]
    cql_z_filtered = cql_z[bcq_unmasked_mask]
    bcq_z_filtered = bcq_z[bcq_unmasked_mask]

    z_scores_unmasked = np.vstack([dqn_z_filtered, bc_z_filtered, cql_z_filtered, bcq_z_filtered])
    z_corr_unmasked = np.corrcoef(z_scores_unmasked)

    print(f"\nCorrelation Matrix - Unmasked BCQ Samples Only ({len(bcq_z_filtered)}/{len(bcq_z)} samples):")
    print(f"{'':>12s}", end='')
    for name in z_names:
        print(f"{name:>12s}", end='')
    print()
    print("-"*60)

    for i, name in enumerate(z_names):
        print(f"{name:>12s}", end='')
        for j in range(len(z_names)):
            print(f"{z_corr_unmasked[i, j]:>12.4f}", end='')
        print()

    # Analysis: How do other models' Z-scores differ between masked vs unmasked BCQ samples?
    print(f"\nDiagnostic Analysis - Why does masking affect correlation?")
    print("-"*80)
    print(f"Number of masked samples: {bcq_masked_mask.sum():,} ({bcq_masked_mask.sum()/len(bcq_z)*100:.2f}%)")
    print(f"Number of unmasked samples: {bcq_unmasked_mask.sum():,} ({bcq_unmasked_mask.sum()/len(bcq_z)*100:.2f}%)")

    print(f"\nZ-scores for MASKED BCQ samples:")
    print(f"  Z_DQN:  mean={np.mean(dqn_z[bcq_masked_mask]):.4f}, std={np.std(dqn_z[bcq_masked_mask]):.4f}")
    print(f"  Z_BC:   mean={np.mean(bc_z[bcq_masked_mask]):.4f}, std={np.std(bc_z[bcq_masked_mask]):.4f}")
    print(f"  Z_CQL:  mean={np.mean(cql_z[bcq_masked_mask]):.4f}, std={np.std(cql_z[bcq_masked_mask]):.4f}")

    print(f"\nZ-scores for UNMASKED BCQ samples:")
    print(f"  Z_DQN:  mean={np.mean(dqn_z[bcq_unmasked_mask]):.4f}, std={np.std(dqn_z[bcq_unmasked_mask]):.4f}")
    print(f"  Z_BC:   mean={np.mean(bc_z[bcq_unmasked_mask]):.4f}, std={np.std(bc_z[bcq_unmasked_mask]):.4f}")
    print(f"  Z_CQL:  mean={np.mean(cql_z[bcq_unmasked_mask]):.4f}, std={np.std(cql_z[bcq_unmasked_mask]):.4f}")

    print("\nCognitive Indices Statistics:")
    print(f"  I_habit     (BC - DQN):  mean={np.mean(I_habit):.4f}, std={np.std(I_habit):.4f}, "
          f"min={np.min(I_habit):.4f}, max={np.max(I_habit):.4f}")
    print(f"  I_pessimism (CQL - DQN): mean={np.mean(I_pessimism):.4f}, std={np.std(I_pessimism):.4f}, "
          f"min={np.min(I_pessimism):.4f}, max={np.max(I_pessimism):.4f}")
    print(f"  I_prudence  (BCQ - DQN): mean={np.mean(I_prudence):.4f}, std={np.std(I_prudence):.4f}, "
          f"min={np.min(I_prudence):.4f}, max={np.max(I_prudence):.4f}")

    # Compute correlations between cognitive indices (for multicollinearity check)
    print("\n" + "="*80)
    print("COGNITIVE INDICES CORRELATIONS (Multicollinearity Check)")
    print("="*80)

    # Create correlation matrix
    indices = np.vstack([I_habit, I_pessimism, I_prudence])
    corr_matrix = np.corrcoef(indices)

    # Print correlation matrix
    index_names = ['I_habit', 'I_pessimism', 'I_prudence']
    print("\nCorrelation Matrix (Pearson):")
    print(f"{'':>15s}", end='')
    for name in index_names:
        print(f"{name:>15s}", end='')
    print()
    print("-"*60)

    for i, name in enumerate(index_names):
        print(f"{name:>15s}", end='')
        for j in range(len(index_names)):
            print(f"{corr_matrix[i, j]:>15.4f}", end='')
        print()

    # Print pairwise correlations
    print("\nPairwise Correlations:")
    print(f"  I_habit vs I_pessimism:    r = {corr_matrix[0, 1]:>7.4f}")
    print(f"  I_habit vs I_prudence:     r = {corr_matrix[0, 2]:>7.4f}")
    print(f"  I_pessimism vs I_prudence: r = {corr_matrix[1, 2]:>7.4f}")

    # =========================================================================
    # OOD SUBSET ANALYSIS: Bottom 10% of each cognitive index
    # =========================================================================

    print("\n" + "="*80)
    print("OOD SUBSET ANALYSIS: Bottom 10% of Each Cognitive Index")
    print("="*80)

    subsets = [
        ('I_habit (bottom 10%)',     I_habit,     0),
        ('I_pessimism (bottom 10%)', I_pessimism, 1),
        ('I_prudence (bottom 10%)',  I_prudence,  2),
    ]

    for subset_name, index_arr, _ in subsets:
        threshold = np.percentile(index_arr, 10)
        mask = index_arr <= threshold
        n_subset = mask.sum()

        print(f"--- {subset_name} ---")
        print(f"  Threshold (10th pct): {threshold:.4f}")
        print(f"  Subset size: {n_subset:,} / {len(index_arr):,} ({n_subset/len(index_arr)*100:.1f}%)")

        # Index statistics in this subset
        I_h_sub = I_habit[mask]
        I_p_sub = I_pessimism[mask]
        I_pr_sub = I_prudence[mask]

        print(f"  I_habit     in subset: mean={np.mean(I_h_sub):.4f}, std={np.std(I_h_sub):.4f}")
        print(f"  I_pessimism in subset: mean={np.mean(I_p_sub):.4f}, std={np.std(I_p_sub):.4f}")
        print(f"  I_prudence  in subset: mean={np.mean(I_pr_sub):.4f}, std={np.std(I_pr_sub):.4f}")

        # Cognitive index correlation matrix in this subset
        sub_indices = np.vstack([I_h_sub, I_p_sub, I_pr_sub])
        sub_corr = np.corrcoef(sub_indices)

        print(f"\n  Cognitive Index Correlation Matrix (subset):")
        print(f"  {'':>15s}", end='')
        for name in index_names:
            print(f"{name:>15s}", end='')
        print()
        print("  " + "-"*60)
        for i, name in enumerate(index_names):
            print(f"  {name:>15s}", end='')
            for j in range(len(index_names)):
                print(f"{sub_corr[i, j]:>15.4f}", end='')
            print()

        # Z-score correlation matrix in this subset
        dqn_z_sub  = dqn_z[mask]
        bc_z_sub   = bc_z[mask]
        cql_z_sub  = cql_z[mask]
        bcq_z_sub  = bcq_z[mask]

        sub_z = np.vstack([dqn_z_sub, bc_z_sub, cql_z_sub, bcq_z_sub])
        sub_z_corr = np.corrcoef(sub_z)

        print(f"\n  Z-Score Correlation Matrix (subset):")
        print(f"  {'':>12s}", end='')
        for name in z_names:
            print(f"{name:>12s}", end='')
        print()
        print("  " + "-"*52)
        for i, name in enumerate(z_names):
            print(f"  {name:>12s}", end='')
            for j in range(len(z_names)):
                print(f"{sub_z_corr[i, j]:>12.4f}", end='')
            print()

        # Pairwise index correlations (concise)
        print(f"\n  Pairwise Index Correlations (subset vs full):")
        pairs = [
            ('I_habit vs I_pessimism',    0, 1),
            ('I_habit vs I_prudence',     0, 2),
            ('I_pessimism vs I_prudence', 1, 2),
        ]
        for pair_name, i, j in pairs:
            print(f"    {pair_name:30s}: subset r={sub_corr[i,j]:>7.4f}  |  full r={corr_matrix[i,j]:>7.4f}  |  Δ={sub_corr[i,j]-corr_matrix[i,j]:>+7.4f}")

        print()

    print("="*80)
    print("DONE")
    print("="*80)


if __name__ == '__main__':
    main()
