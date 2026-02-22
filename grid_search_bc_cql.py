"""
grid_search_bc_cql.py

Evaluate BC, CQL, and a combined model on the test set.

Combined model:
    score = β * Q_CQL + λ * log(π_BC)
    π = softmax(score)

Grid search over β ∈ [0.01, 0.1, 1, 10] × λ ∈ [0.01, 0.1, 1, 10]
to find the lowest CE (= best alignment with human behavior).

Usage:
    python grid_search_bc_cql.py \\
        --subject sub_1 \\
        --bc-path  pretrained/bc.pth \\
        --cql-path pretrained/cql.pth
"""

import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import create_train_val_dataloaders
from model import BehaviorCloning, CQL, load_pretrained_cnn


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description='BC + CQL combined model grid search')

    # Data
    parser.add_argument('--data-dir', type=str,
                        default='/Users/seokwon/research/fMRI_RL/processed_data_frameskip_4',
                        help='Base directory containing processed data')
    parser.add_argument('--subject', type=str, default='sub_1',
                        choices=['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6'])
    parser.add_argument('--val-file-idx', type=int, default=9,
                        help='Index of validation file (0-based)')
    parser.add_argument('--test-file-idx', type=int, default=10,
                        help='Index of test file (0-based)')

    # Model paths
    parser.add_argument('--dqn-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/pretrained/dqn_cnn.pt',
                        help='Path to pretrained DQN CNN encoder')
    parser.add_argument('--bc-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/pretrained/bc.pth',
                        help='Path to trained BC model')
    parser.add_argument('--cql-path', type=str,
                        default='/Users/seokwon/research/fMRI_RL/pretrained/cql.pth',
                        help='Path to trained CQL model')

    # Misc
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_logits(bc, cql, loader, device):
    """
    Single forward pass through all test data.

    Returns (on CPU):
        bc_logits : (N, 6)  – raw logits from BC
        q_values  : (N, 6)  – raw Q-values from CQL
        actions   : (N,)    – human action indices
    """
    bc.eval()
    cql.eval()

    all_bc, all_q, all_a = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Collecting logits', ncols=80):
            state  = batch['state'].to(device).float() / 255.0
            action = batch['action'].to(device)
            if action.dim() == 2 and action.size(1) > 1:
                action = action.argmax(dim=-1)

            all_bc.append(bc(state).cpu())
            all_q.append(cql(state).cpu())
            all_a.append(action.cpu())

    return (
        torch.cat(all_bc, dim=0),
        torch.cat(all_q,  dim=0),
        torch.cat(all_a,  dim=0).long(),
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def topk_acc(logits, actions, k):
    """Fraction of samples where correct action is in top-k predicted actions."""
    topk_idx = logits.topk(k, dim=-1).indices  # (N, k)
    return (topk_idx == actions.unsqueeze(1)).any(dim=1).float().mean().item()


def eval_bc(bc_logits, actions):
    """CE and top-1/2/3 accuracy using raw BC logits."""
    ce   = F.cross_entropy(bc_logits, actions).item()
    acc1 = topk_acc(bc_logits, actions, 1)
    acc2 = topk_acc(bc_logits, actions, 2)
    acc3 = topk_acc(bc_logits, actions, 3)
    return ce, acc1, acc2, acc3


def z_normalize_q(q_values):
    """Per-sample z-score normalisation of Q-values (matches val_cql)."""
    q_mean = q_values.mean(dim=-1, keepdim=True)
    q_std  = q_values.std(dim=-1, keepdim=True) + 1e-8
    return (q_values - q_mean) / q_std


def eval_cql(z_q, actions):
    """CE and top-1/2/3 accuracy using z-score normalised Q-values (matches val_cql)."""
    ce   = F.cross_entropy(z_q, actions).item()
    acc1 = topk_acc(z_q, actions, 1)
    acc2 = topk_acc(z_q, actions, 2)
    acc3 = topk_acc(z_q, actions, 3)
    return ce, acc1, acc2, acc3


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(bc_logits, z_q, actions, betas, lambdas):
    """
    score = β * z_Q + λ * log(π_BC)   (z_Q = per-sample z-score normalised Q)
    CE    = cross_entropy(score, actions)
    Returns list of (beta, lam, ce, acc1, acc2, acc3) sorted by CE ascending (lower = better).
    """
    log_pi_bc = F.log_softmax(bc_logits, dim=-1)

    results = []
    for beta in betas:
        for lam in lambdas:
            score = beta * z_q + lam * log_pi_bc
            ce    = F.cross_entropy(score, actions).item()
            acc1  = topk_acc(score, actions, 1)
            acc2  = topk_acc(score, actions, 2)
            acc3  = topk_acc(score, actions, 3)
            results.append((beta, lam, ce, acc1, acc2, acc3))

    results.sort(key=lambda x: x[2])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = get_args()
    device = torch.device(args.device)

    print("=" * 80)
    print("BC + CQL COMBINED MODEL GRID SEARCH")
    print("=" * 80)
    print(f"Subject         : {args.subject}")
    print(f"Val  file idx   : {args.val_file_idx}")
    print(f"Test file idx   : {args.test_file_idx}")
    print(f"BC path         : {args.bc_path}")
    print(f"CQL path        : {args.cql_path}")
    print(f"Device          : {device}")
    print("=" * 80)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nLoading data...")
    _, _, test_loader = create_train_val_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        subject=args.subject,
        num_workers=args.num_workers,
        val_file_idx=args.val_file_idx,
        test_file_idx=args.test_file_idx,
    )
    n_test = len(test_loader.dataset)
    print(f"✓ Test batches: {len(test_loader)}  ({n_test:,} samples)")

    # ── Models ────────────────────────────────────────────────────────────
    print("\nLoading pretrained CNN...")
    cnn_bc  = load_pretrained_cnn(args.dqn_path, freeze=True).to(device)
    cnn_cql = load_pretrained_cnn(args.dqn_path, freeze=True).to(device)

    print("Loading BC model...")
    bc = BehaviorCloning(cnn_bc, action_dim=6)
    bc.load_state_dict(torch.load(args.bc_path, map_location=device))
    bc = bc.to(device).eval()
    print("✓ BC loaded")

    print("Loading CQL model...")
    cql = CQL(cnn_cql, action_dim=6)
    cql.load_state_dict(torch.load(args.cql_path, map_location=device))
    cql = cql.to(device).eval()
    print("✓ CQL loaded")

    # ── Single forward pass ───────────────────────────────────────────────
    print("\nRunning forward pass on test set...")
    bc_logits, q_values, actions = collect_logits(bc, cql, test_loader, device)
    print(f"✓ bc_logits  : {tuple(bc_logits.shape)}   Q range [{q_values.min():.2f}, {q_values.max():.2f}]")

    # z-score normalise Q once — used for both CQL baseline and combined model
    z_q = z_normalize_q(q_values)

    # ── Baselines ─────────────────────────────────────────────────────────
    bc_ce,  bc_acc1,  bc_acc2,  bc_acc3  = eval_bc(bc_logits, actions)
    cql_ce, cql_acc1, cql_acc2, cql_acc3 = eval_cql(z_q, actions)

    print("\n" + "=" * 90)
    print("BASELINE RESULTS  (test set, lower CE = better)")
    print("=" * 90)
    print(f"{'Model':<25s} {'CE ↓':<12s} {'Top-1':<10s} {'Top-2':<10s} {'Top-3'}")
    print("-" * 67)
    print(f"{'BC':<25s} {bc_ce:<12.4f} {bc_acc1*100:<10.2f} {bc_acc2*100:<10.2f} {bc_acc3*100:.2f}%")
    print(f"{'CQL  (z-norm Q)':<25s} {cql_ce:<12.4f} {cql_acc1*100:<10.2f} {cql_acc2*100:<10.2f} {cql_acc3*100:.2f}%")

    # ── Grid search ───────────────────────────────────────────────────────
    betas = [
        -3.0, -1.0, -0.7, -0.5, -0.3, -0.1, -0.07, -0.05, -0.03, -0.01,
        0.0,
        0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0
    ]

    lambdas = [
        -1.0, -0.7, -0.5, -0.3, -0.1, -0.07, -0.05, -0.03, -0.01,
        0.0,
        0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0
    ]

    print("\n" + "=" * 90)
    print(f"GRID SEARCH   score = β·z_Q + λ·log(π_BC)   "
          f"({len(betas)}×{len(lambdas)} = {len(betas)*len(lambdas)} configs)")
    print("=" * 90)

    results = grid_search(bc_logits, z_q, actions, betas, lambdas)

    print(f"\n{'β':<10s} {'λ':<10s} {'CE ↓':<12s} {'Top-1':<10s} {'Top-2':<10s} {'Top-3':<10s}")
    print("-" * 62)
    for i, (beta, lam, ce, acc1, acc2, acc3) in enumerate(results):
        tag = " ← BEST" if i == 0 else ""
        print(f"{beta:<10.2f} {lam:<10.2f} {ce:<12.4f} {acc1*100:<10.2f} {acc2*100:<10.2f} {acc3*100:<10.2f}{tag}")

    best_beta, best_lam, best_ce, best_acc1, best_acc2, best_acc3 = results[0]

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY  (test set)")
    print("=" * 90)
    print(f"{'Model':<35s} {'CE ↓':<12s} {'Top-1':<10s} {'Top-2':<10s} {'Top-3'}")
    print("-" * 77)
    print(f"{'BC':<35s} {bc_ce:<12.4f} {bc_acc1*100:<10.2f} {bc_acc2*100:<10.2f} {bc_acc3*100:.2f}%")
    print(f"{'CQL  (z-norm Q)':<35s} {cql_ce:<12.4f} {cql_acc1*100:<10.2f} {cql_acc2*100:<10.2f} {cql_acc3*100:.2f}%")
    print(f"{'Combined  β={:.2f} λ={:.2f}'.format(best_beta, best_lam):<35s} "
          f"{best_ce:<12.4f} {best_acc1*100:<10.2f} {best_acc2*100:<10.2f} {best_acc3*100:.2f}%")
    print("=" * 90)
    print(f"\nCombined vs BC :   ΔCE = {best_ce - bc_ce:+.4f}  ΔTop1 = {(best_acc1 - bc_acc1)*100:+.2f}%")
    print(f"Combined vs CQL:   ΔCE = {best_ce - cql_ce:+.4f}  ΔTop1 = {(best_acc1 - cql_acc1)*100:+.2f}%")
    print("=" * 90)


if __name__ == '__main__':
    main()
