import torch
import torch.optim as optim
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

from args import get_args
from dataset import create_train_val_dataloaders, create_all_data_dataloader
from model import (
    load_pretrained_cnn,
    BehaviorCloning, train_bc, val_bc,
    BCQ, train_bcq, val_bcq,
    CQL, train_cql, val_cql,
    IQL, train_iql, val_iql,
    ProbCQL, train_prob_cql, val_prob_cql,
    EnsembleDQN, train_ensemble_dqn, val_ensemble_dqn,
)
from eval import evaluate_agent


def safe_save(obj, path: Path):
    """원자적 저장: tmp 파일에 먼저 쓴 뒤 rename해 EOCD 누락 방지."""
    path = Path(path)
    tmp  = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_action_weights(dataloader, device='cpu', exponent=0.5):
    """Compute 6-class inverse-frequency weights for cross-entropy loss."""
    counts = torch.zeros(6, dtype=torch.long)
    for batch in dataloader:
        action = batch['action']
        if action.dim() == 2:
            action_idx = action.argmax(dim=-1)
        else:
            action_idx = action.long()
        for a in action_idx:
            counts[a] += 1

    total = counts.sum().float()
    if exponent == 0.0:
        weights = torch.ones(6)
    else:
        weights = (total / (6 * counts.float().clamp_min(1.0))) ** exponent

    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHT+FIRE', 'LEFT+FIRE']
    print("\n6-class Action Distribution:")
    for i in range(6):
        pct = (counts[i].float() / total * 100).item()
        print(f"  {action_names[i]:>12s}: {counts[i]:6,} ({pct:5.2f}%) - Weight: {weights[i]:.4f}")

    return weights.to(device)


def compute_fire_move_weights(dataloader, device='cpu', exponent=0.5):
    from model.bc import action_to_fire_move

    print(f"\nComputing fire/move class weights from action distribution (exponent={exponent})...")
    fire_counts = torch.zeros(2, dtype=torch.long)  # fire=0, fire=1
    move_counts = torch.zeros(3, dtype=torch.long)  # move=0, move=1, move=2

    for batch in dataloader:
        action = batch['action']

        # Convert one-hot to index if needed
        if action.dim() == 2:
            action_idx = action.argmax(dim=-1)
        else:
            action_idx = action

        # Convert to fire and move labels
        fire_label, move_label = action_to_fire_move(action_idx)

        # Count each fire and move
        for f in fire_label:
            fire_counts[f] += 1
        for m in move_label:
            move_counts[m] += 1

    total_samples = fire_counts.sum().float()

    # Compute weights
    if exponent == 0.0:
        # No weighting - all weights are 1
        fire_weights = torch.ones(2)
        move_weights = torch.ones(3)
    else:
        # Prevent division by zero with clamp_min
        fire_counts_safe = fire_counts.float().clamp_min(1.0)
        move_counts_safe = move_counts.float().clamp_min(1.0)

        # Apply exponent to inverse frequency weights
        fire_raw_weights = total_samples / (2 * fire_counts_safe)
        fire_weights = fire_raw_weights ** exponent

        move_raw_weights = total_samples / (3 * move_counts_safe)
        move_weights = move_raw_weights ** exponent

    # Print distribution
    print("\nFire Distribution:")
    fire_names = ['No Fire (NOOP, RIGHT, LEFT)', 'Fire (FIRE, RIGHT+FIRE, LEFT+FIRE)']
    for i in range(2):
        percentage = (fire_counts[i].float() / total_samples * 100).item()
        print(f"  {fire_names[i]:40s}: {fire_counts[i]:6,} ({percentage:5.2f}%) - Weight: {fire_weights[i]:.4f}")

    print("\nMove Distribution:")
    move_names = ['No Move (NOOP, FIRE)', 'Right (RIGHT, RIGHT+FIRE)', 'Left (LEFT, LEFT+FIRE)']
    for i in range(3):
        percentage = (move_counts[i].float() / total_samples * 100).item()
        print(f"  {move_names[i]:40s}: {move_counts[i]:6,} ({percentage:5.2f}%) - Weight: {move_weights[i]:.4f}")

    return fire_weights.to(device), move_weights.to(device)


def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir) / args.algo
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models will be saved to: {save_dir}")

    print("\nLoading data...")
    if args.all_data:
        all_loader = create_all_data_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            subject=args.subject,
            num_workers=args.num_workers,
        )
        train_loader = val_loader = test_loader = all_loader
        print(f"All-data mode: {len(all_loader)} batches per epoch (train = val = test)")
    else:
        train_loader, val_loader, test_loader = create_train_val_dataloaders(
            args.data_dir,
            batch_size=args.batch_size,
            subject=args.subject,
            num_workers=args.num_workers,
            val_file_idx=args.val_file_idx,
            test_file_idx=args.test_file_idx,
        )
        print(f"Train batches per epoch: {len(train_loader)}")
        if val_loader  is not None: print(f"Val batches            : {len(val_loader)}")
        if test_loader is not None: print(f"Test batches           : {len(test_loader)}")

    print("\nLoading pretrained DQN CNN (frozen)...")
    cnn = load_pretrained_cnn(args.dqn_path, freeze=True)
    cnn = cnn.to(device)

    # Create model based on algorithm
    print(f"\nCreating {args.algo.upper()} model...")

    # Compute 6-class action weights (used by all algos for val CE)
    if args.class_weight_exponent > 0:
        action_weights = compute_action_weights(train_loader, device, args.class_weight_exponent)
    else:
        action_weights = None

    if args.algo == 'bc':
        model = BehaviorCloning(cnn, action_dim=6)
        train_fn = train_bc
        val_fn = val_bc
        print("BC Loss: Fire Binary CE + Move 3-class CE")

        # Compute fire/move class weights if weight exponent > 0
        if args.class_weight_exponent > 0:
            fire_weights, move_weights = compute_fire_move_weights(
                train_loader, device, args.class_weight_exponent
            )
        else:
            fire_weights, move_weights = None, None
            print("\nNo class weighting (exponent=0.0)")

    elif args.algo == 'bcq':
        model = BCQ(cnn, action_dim=6, threshold=args.bcq_threshold, bc_path=args.bc_path)
        train_fn = train_bcq
        val_fn = val_bcq

    elif args.algo == 'cql':
        model = CQL(cnn, action_dim=6, alpha=args.cql_alpha)
        train_fn = train_cql
        val_fn = val_cql

    elif args.algo == 'prob_cql':
        model = ProbCQL(cnn, action_dim=6, alpha=args.cql_alpha,
                        beta_kl=args.prob_cql_beta_kl)
        train_fn = train_prob_cql
        val_fn = val_prob_cql
        print(f"ProbCQL: α={args.cql_alpha}, β_kl={args.prob_cql_beta_kl}")

    elif args.algo == 'ensemble_dqn':
        model = EnsembleDQN(cnn, num_heads=args.num_heads, action_dim=6,
                            temp=args.ens_temp, agg=args.ens_agg)
        train_fn = train_ensemble_dqn
        val_fn = val_ensemble_dqn
        print(f"EnsembleDQN: {args.num_heads} heads, mask_prob={args.mask_prob}, "
              f"temp={args.ens_temp}, agg={args.ens_agg}")

    elif args.algo == 'iql':
        model = IQL(cnn, action_dim=6, tau=args.iql_tau, beta=args.iql_beta,
                    max_weight=args.iql_max_weight, target_update_freq=args.target_update_freq)
        print(f"IQL: τ={args.iql_tau}, β={args.iql_beta}, max_weight={args.iql_max_weight}")

    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    model = model.to(device)

    # Optimizer (encoder is always frozen, only optimize trainable parameters)
    if args.algo == 'iql':
        q_optimizer     = optim.Adam(model.q_network.parameters(), lr=args.lr)
        v_optimizer     = optim.Adam(model.v_network.parameters(), lr=args.lr)
        actor_optimizer = optim.Adam(model.actor.parameters(),     lr=args.lr)
        optimizer = None  # not used for IQL
        print(f"IQL Optimizers: 3× Adam LR={args.lr:.2e} (q, v, actor)")
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr
        )
        print(f"Optimizer: LR={args.lr:.2e} (encoder frozen)")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*80)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch"):
        # ===== TRAINING =====
        if args.algo == 'bc':
            train_loss, train_acc, train_fire_loss, train_move_loss, train_fire_acc, train_move_acc = train_fn(
                model, train_loader, optimizer, device, args.label_smoothing, fire_weights, move_weights,
                args.fire_loss_weight, args.move_loss_weight
            )

            log = (
                f"Epoch {epoch}/{args.epochs}\n"
                f"  Train - Loss: {train_loss:.4f}, Action Acc: {train_acc:.4f}\n"
                f"          Fire Loss: {train_fire_loss:.4f}, Fire Acc: {train_fire_acc:.4f}\n"
                f"          Move Loss: {train_move_loss:.4f}, Move Acc: {train_move_acc:.4f}"
            )
            if not args.all_data:
                if val_loader is not None:
                    val_loss, val_acc, val_fire_loss, val_move_loss, val_fire_acc, val_move_acc, val_ce, val_wce = val_fn(
                        model, val_loader, device, args.label_smoothing, fire_weights, move_weights,
                        args.fire_loss_weight, args.move_loss_weight, action_weights
                    )
                    log += (
                        f"\n  Val   - Loss: {val_loss:.4f}, Action Acc: {val_acc:.4f}\n"
                        f"          Fire Loss: {val_fire_loss:.4f}, Fire Acc: {val_fire_acc:.4f}\n"
                        f"          Move Loss: {val_move_loss:.4f}, Move Acc: {val_move_acc:.4f}\n"
                        f"          CE: {val_ce:.4f}, Weighted CE: {val_wce:.4f}"
                    )
                if test_loader is not None:
                    test_loss, test_acc, test_fire_loss, test_move_loss, test_fire_acc, test_move_acc, test_ce, test_wce = val_fn(
                        model, test_loader, device, args.label_smoothing, fire_weights, move_weights,
                        args.fire_loss_weight, args.move_loss_weight, action_weights
                    )
                    log += (
                        f"\n  Test  - Loss: {test_loss:.4f}, Action Acc: {test_acc:.4f}\n"
                        f"          Fire Loss: {test_fire_loss:.4f}, Fire Acc: {test_fire_acc:.4f}\n"
                        f"          Move Loss: {test_move_loss:.4f}, Move Acc: {test_move_acc:.4f}\n"
                        f"          CE: {test_ce:.4f}, Weighted CE: {test_wce:.4f}"
                    )
            tqdm.write(log)

            # Save model every save_interval epochs (safe_save로 EOCD 누락 방지)
            if epoch % args.save_interval == 0:
                safe_save(model.state_dict(), save_dir / f'epoch_{epoch}.pth')
                tqdm.write(f"  ✓ Saved model: epoch_{epoch}.pth")

        elif args.algo == 'bcq':
            train_q_loss, train_avg_q = train_fn(
                model, train_loader, optimizer, device, args.gamma, args.target_update_freq, args.reward_scale
            )

            log = (
                f"Epoch {epoch}/{args.epochs}\n"
                f"  Train - Q Loss: {train_q_loss:.4f}, Avg Q: {train_avg_q:.2f}"
            )
            if not args.all_data:
                if val_loader is not None:
                    val_q_loss, val_avg_q, val_ce, val_wce, val_acc = val_fn(
                        model, val_loader, device, args.gamma, args.reward_scale, action_weights
                    )
                    log += (
                        f"\n  Val   - Q Loss: {val_q_loss:.4f}, Avg Q: {val_avg_q:.2f}\n"
                        f"          CE: {val_ce:.4f}, Weighted CE: {val_wce:.4f}, Action Acc: {val_acc:.4f}"
                    )
                if test_loader is not None:
                    test_q_loss, test_avg_q, test_ce, test_wce, test_acc = val_fn(
                        model, test_loader, device, args.gamma, args.reward_scale, action_weights
                    )
                    log += (
                        f"\n  Test  - Q Loss: {test_q_loss:.4f}, Avg Q: {test_avg_q:.2f}\n"
                        f"          CE: {test_ce:.4f}, Weighted CE: {test_wce:.4f}, Action Acc: {test_acc:.4f}"
                    )
            tqdm.write(log)

            # Save model every save_interval epochs (safe_save로 EOCD 누락 방지)
            if epoch % args.save_interval == 0:
                safe_save(model.state_dict(), save_dir / f'epoch_{epoch}.pth')
                tqdm.write(f"  ✓ Saved model: epoch_{epoch}.pth")

        elif args.algo == 'cql':
            train_td_loss, train_cql_loss, train_total_loss, train_avg_q = train_fn(
                model, train_loader, optimizer, device, args.gamma, args.target_update_freq, args.reward_scale
            )

            log = (
                f"Epoch {epoch}/{args.epochs}\n"
                f"  Train - TD Loss: {train_td_loss:.4f}, CQL Loss: {train_cql_loss:.4f}, "
                f"Total: {train_total_loss:.4f}, Avg Q: {train_avg_q:.2f}"
            )
            if not args.all_data:
                if val_loader is not None:
                    val_td_loss, val_cql_loss, val_total_loss, val_avg_q, val_ce, val_wce, val_acc = val_fn(
                        model, val_loader, device, args.gamma, args.reward_scale, action_weights
                    )
                    log += (
                        f"\n  Val   - TD Loss: {val_td_loss:.4f}, CQL Loss: {val_cql_loss:.4f}, "
                        f"Total: {val_total_loss:.4f}, Avg Q: {val_avg_q:.2f}\n"
                        f"          CE: {val_ce:.4f}, Weighted CE: {val_wce:.4f}, Action Acc: {val_acc:.4f}"
                    )
                if test_loader is not None:
                    test_td_loss, test_cql_loss, test_total_loss, test_avg_q, test_ce, test_wce, test_acc = val_fn(
                        model, test_loader, device, args.gamma, args.reward_scale, action_weights
                    )
                    log += (
                        f"\n  Test  - TD Loss: {test_td_loss:.4f}, CQL Loss: {test_cql_loss:.4f}, "
                        f"Total: {test_total_loss:.4f}, Avg Q: {test_avg_q:.2f}\n"
                        f"          CE: {test_ce:.4f}, Weighted CE: {test_wce:.4f}, Action Acc: {test_acc:.4f}"
                    )
            tqdm.write(log)

            # Save model every save_interval epochs (safe_save로 EOCD 누락 방지)
            if epoch % args.save_interval == 0:
                safe_save(model.state_dict(), save_dir / f'epoch_{epoch}.pth')
                tqdm.write(f"  ✓ Saved model: epoch_{epoch}.pth")

        elif args.algo == 'prob_cql':
            (train_td, train_cql_loss, train_kl, train_total,
             train_avg_q, train_avg_var) = train_fn(
                model, train_loader, optimizer, device,
                args.gamma, args.target_update_freq, args.reward_scale
            )

            log = (
                f"Epoch {epoch}/{args.epochs}\n"
                f"  Train - TD: {train_td:.4f}, CQL: {train_cql_loss:.4f}, KL: {train_kl:.4f}, "
                f"Total: {train_total:.4f}, Q: {train_avg_q:.2f}, Var: {train_avg_var:.4f}"
            )
            if not args.all_data:
                if val_loader is not None:
                    (val_td, val_cql_loss, val_kl, val_total,
                     val_avg_q, val_avg_var,
                     val_ce, val_wce, val_acc) = val_fn(
                        model, val_loader, device,
                        args.gamma, args.reward_scale, action_weights
                    )
                    log += (
                        f"\n  Val   - TD: {val_td:.4f}, CQL: {val_cql_loss:.4f}, KL: {val_kl:.4f}, "
                        f"Total: {val_total:.4f}, Q: {val_avg_q:.2f}, Var: {val_avg_var:.4f}\n"
                        f"          CE: {val_ce:.4f}, Weighted CE: {val_wce:.4f}, Action Acc: {val_acc:.4f}"
                    )
                if test_loader is not None:
                    (test_td, test_cql_loss, test_kl, test_total,
                     test_avg_q, test_avg_var,
                     test_ce, test_wce, test_acc) = val_fn(
                        model, test_loader, device,
                        args.gamma, args.reward_scale, action_weights
                    )
                    log += (
                        f"\n  Test  - TD: {test_td:.4f}, CQL: {test_cql_loss:.4f}, KL: {test_kl:.4f}, "
                        f"Total: {test_total:.4f}, Q: {test_avg_q:.2f}, Var: {test_avg_var:.4f}\n"
                        f"          CE: {test_ce:.4f}, Weighted CE: {test_wce:.4f}, Action Acc: {test_acc:.4f}"
                    )
            tqdm.write(log)

            # Save model every save_interval epochs (safe_save로 EOCD 누락 방지)
            if epoch % args.save_interval == 0:
                safe_save(model.state_dict(), save_dir / f'epoch_{epoch}.pth')
                tqdm.write(f"  ✓ Saved model: epoch_{epoch}.pth")

        elif args.algo == 'ensemble_dqn':
            train_td_loss, train_ce_loss, train_total_loss, train_avg_q = train_fn(
                model, train_loader, optimizer, device,
                args.gamma, args.target_update_freq, args.reward_scale, args.mask_prob,
                args.ce_lambda
            )

            log = (
                f"Epoch {epoch}/{args.epochs}\n"
                f"  Train - TD: {train_td_loss:.4f}, CE: {train_ce_loss:.4f}, "
                f"Total: {train_total_loss:.4f}, Avg Q: {train_avg_q:.2f}"
            )
            if not args.all_data:
                if val_loader is not None:
                    val_td_loss, val_avg_q, val_ce, val_wce, val_acc = val_fn(
                        model, val_loader, device, args.gamma, args.reward_scale, action_weights
                    )
                    log += (
                        f"\n  Val   - TD: {val_td_loss:.4f}, Avg Q: {val_avg_q:.2f}\n"
                        f"          CE: {val_ce:.4f}, Weighted CE: {val_wce:.4f}, Action Acc: {val_acc:.4f}"
                    )
                if test_loader is not None:
                    test_td_loss, test_avg_q, test_ce, test_wce, test_acc = val_fn(
                        model, test_loader, device, args.gamma, args.reward_scale, action_weights
                    )
                    log += (
                        f"\n  Test  - TD: {test_td_loss:.4f}, Avg Q: {test_avg_q:.2f}\n"
                        f"          CE: {test_ce:.4f}, Weighted CE: {test_wce:.4f}, Action Acc: {test_acc:.4f}"
                    )
            tqdm.write(log)

            # Save heads only (CNN is always the same pretrained encoder)
            if epoch % args.save_interval == 0:
                safe_save(model.heads.state_dict(), save_dir / f'epoch_{epoch}.pth')
                tqdm.write(f"  ✓ Saved model: epoch_{epoch}.pth")

        elif args.algo == 'iql':
            (train_q_loss, train_v_loss, train_actor_loss,
             train_avg_q, train_avg_v, train_avg_adv, train_avg_w) = train_iql(
                model, train_loader, q_optimizer, v_optimizer, actor_optimizer,
                device, args.gamma, args.target_update_freq, args.reward_scale
            )

            log = (
                f"Epoch {epoch}/{args.epochs}\n"
                f"  Train - Q: {train_q_loss:.4f}, V: {train_v_loss:.4f}, "
                f"Actor: {train_actor_loss:.4f}, Avg Q: {train_avg_q:.2f}, "
                f"Avg V: {train_avg_v:.2f}, Adv: {train_avg_adv:.2f}, W: {train_avg_w:.2f}"
            )
            if not args.all_data:
                if val_loader is not None:
                    (val_q_loss, val_v_loss, val_actor_loss,
                     val_avg_q, val_avg_v, val_avg_adv, val_avg_w,
                     val_ce, val_wce, val_acc) = val_iql(
                        model, val_loader, device, args.gamma, args.reward_scale, action_weights
                    )
                    log += (
                        f"\n  Val   - Q: {val_q_loss:.4f}, V: {val_v_loss:.4f}, "
                        f"Actor: {val_actor_loss:.4f}, Avg Q: {val_avg_q:.2f}, "
                        f"Avg V: {val_avg_v:.2f}, Adv: {val_avg_adv:.2f}, W: {val_avg_w:.2f}\n"
                        f"          CE: {val_ce:.4f}, Weighted CE: {val_wce:.4f}, Action Acc: {val_acc:.4f}"
                    )
                if test_loader is not None:
                    (test_q_loss, test_v_loss, test_actor_loss,
                     test_avg_q, test_avg_v, test_avg_adv, test_avg_w,
                     test_ce, test_wce, test_acc) = val_iql(
                        model, test_loader, device, args.gamma, args.reward_scale, action_weights
                    )
                    log += (
                        f"\n  Test  - Q: {test_q_loss:.4f}, V: {test_v_loss:.4f}, "
                        f"Actor: {test_actor_loss:.4f}, Avg Q: {test_avg_q:.2f}, "
                        f"Avg V: {test_avg_v:.2f}, Adv: {test_avg_adv:.2f}, W: {test_avg_w:.2f}\n"
                        f"          CE: {test_ce:.4f}, Weighted CE: {test_wce:.4f}, Action Acc: {test_acc:.4f}"
                    )
            tqdm.write(log)

            if epoch % args.save_interval == 0:
                safe_save(model.state_dict(), save_dir / f'epoch_{epoch}.pth')
                tqdm.write(f"  ✓ Saved model: epoch_{epoch}.pth")

        # ===== EVALUATION =====
        if epoch % args.eval_interval == 0:
            tqdm.write(f"\n{'='*80}")
            tqdm.write(f"EVALUATION at Epoch {epoch}")
            tqdm.write(f"{'='*80}")

            # Track masking statistics for BCQ
            track_masking = (args.algo == 'bcq')
            eval_stats = evaluate_agent(
                model, args.env_name, device, args.eval_episodes,
                args.seed + epoch, args.deterministic, track_masking=track_masking
            )

            eval_output = (
                f"  Eval - Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}\n"
                f"         Min: {eval_stats['min_reward']:.1f}, Max: {eval_stats['max_reward']:.1f}\n"
                f"         Mean Length: {eval_stats['mean_length']:.1f}"
            )

            # Add masking statistics for BCQ
            if 'mean_allowed_actions' in eval_stats:
                eval_output += (
                    f"\n         BCQ Masking - Mean Allowed: {eval_stats['mean_allowed_actions']:.2f} / 6 actions"
                    f"\n                       Std: {eval_stats['std_allowed_actions']:.2f}"
                )

            tqdm.write(eval_output)
            tqdm.write(f"{'='*80}\n")

    print("\n" + "="*80)
    print("Training complete!")
    print(f"All models saved to: {save_dir}")


if __name__ == '__main__':
    main()
