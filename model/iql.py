import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class IQL(nn.Module):
    """
    Implicit Q-Learning (IQL) for discrete actions
    Architecture: DQN_CNN (frozen, outputs 3136) → three separate heads

    Three heads (all randomly initialized, all trainable):
      Q-network : 3136 → 512 → action_dim   (action-value estimates)
      V-network : 3136 → 512 → 1            (state-value estimate)
      Actor     : 3136 → 512 → action_dim   (categorical policy logits)

    Three losses (IQL, Kostrikov et al. 2021, adapted to discrete actions):
      Q loss   : TD regression  Q(s,a) → r + γ·V(s')
      V loss   : Expectile regression  V(s) ← τ-expectile of Q_target(s,a)
      Actor    : Advantage-weighted behavior cloning
                 L = -mean(w · log π(a_data|s)),  w = exp(β·(Q_target - V))

    Reference: https://arxiv.org/abs/2110.06169
    """

    def __init__(self, cnn, action_dim=6, tau=0.7, beta=3.0,
                 max_weight=100.0, target_update_freq=100):
        """
        Args:
            cnn              : Frozen DQN_CNN backbone (outputs 3136-dim features)
            action_dim       : Number of discrete actions
            tau              : Expectile parameter for V loss (0.5 < tau < 1)
                               Higher tau → V sits closer to the top quantile of Q
            beta             : Temperature for advantage-weighted actor
                               Higher beta → more concentrated imitation of high-adv actions
            max_weight       : Hard upper clamp for exp(beta * adv) weight
            target_update_freq: Steps between hard target Q-network updates
        """
        super(IQL, self).__init__()

        self.action_dim = action_dim
        self.tau = tau
        self.beta = beta
        self.max_weight = max_weight
        self.target_update_freq = target_update_freq

        # CNN backbone (always frozen for stability)
        self.cnn = cnn
        for param in self.cnn.parameters():
            param.requires_grad = False
        print("✓ IQL CNN: All conv layers frozen")

        # Q-network: 3136 → 512 → action_dim
        # Trained by TD regression: Q(s,a) → r + γ·V(s')
        self.q_network = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        # V-network: 3136 → 512 → 1
        # Trained by expectile regression toward Q_target(s, a_data)
        self.v_network = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Actor: 3136 → 512 → action_dim
        # Outputs logits for a categorical policy π(a|s)
        # Trained by advantage-weighted behavior cloning
        self.actor = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        # Target Q-network: stable reference for V loss and Actor loss
        # CNN is shared and frozen; only the Q MLP needs a target copy
        self.q_network_target = copy.deepcopy(self.q_network)
        for param in self.q_network_target.parameters():
            param.requires_grad = False

        # Training step counter (for target network updates)
        self.training_step = 0

    def forward(self, state):
        """Full forward pass (used for analysis / external callers).

        Args:
            state: (B, 4, 84, 84) float in [0, 1]
        Returns:
            q_values : (B, action_dim)
            v_value  : (B, 1)
            logits   : (B, action_dim)
        """
        features = self.cnn(state)           # (B, 3136)
        q_values = self.q_network(features)  # (B, action_dim)
        v_value  = self.v_network(features)  # (B, 1)
        logits   = self.actor(features)      # (B, action_dim)
        return q_values, v_value, logits

    def get_action(self, state, deterministic=True):
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            features = self.cnn(state)
            logits   = self.actor(features)  # (1, action_dim)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                probs  = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)

        return action.item() if action.numel() == 1 else action

    def update_target(self):
        """Hard copy Q-network weights into the target Q-network."""
        self.q_network_target.load_state_dict(self.q_network.state_dict())


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_iql(model, dataloader, q_optimizer, v_optimizer, actor_optimizer,
              device, gamma=0.99, target_update_freq=100, reward_scale=0.1):
    """Train IQL for one epoch.

    Update order (per batch):
        1. V loss  — expectile regression (target is Q_target, stable)
        2. Q loss  — TD regression        (target uses V before this batch's V step;
                                           computed once with no_grad at batch start)
        3. Actor   — advantage-weighted BC (adv uses freshly-stepped V, still detached)
        4. Target Q update (periodic hard copy)

    Rationale for this order:
        - V and Q targets are both derived from Q_target (stable target network),
          so V and Q losses can be computed/applied in either order.
        - Updating V first means the Actor sees the most recent V when computing
          advantages within the same batch, which is slightly fresher.
        - Q target for V loss is computed before the V step (no_grad block),
          so V's update does not affect its own TD target within the batch.

    Args:
        model            : IQL instance
        dataloader       : yields batches with keys:
                             'state'      (B, 4, 84, 84) uint8
                             'action'     (B,) int  or  (B, action_dim) one-hot
                             'reward'     (B,) or (B, 1)
                             'next_state' (B, 4, 84, 84) uint8
                             'done'       (B,) or (B, 1)
        q_optimizer      : optimizer for model.q_network
        v_optimizer      : optimizer for model.v_network
        actor_optimizer  : optimizer for model.actor
        device           : torch device
        gamma            : discount factor
        target_update_freq: hard target update interval (steps)
        reward_scale     : multiply reward by this factor before TD update

    Returns:
        avg_q_loss, avg_v_loss, avg_actor_loss,
        avg_q_value, avg_v_value, avg_adv, avg_weight
    """
    model.train()

    total_q_loss     = 0.0
    total_v_loss     = 0.0
    total_actor_loss = 0.0
    total_q_value    = 0.0
    total_v_value    = 0.0
    total_adv        = 0.0
    total_weight     = 0.0
    total_samples    = 0

    for batch in dataloader:
        model.training_step += 1

        state      = batch['state'].to(device).float() / 255.0        # (B, 4, 84, 84)
        action     = batch['action'].to(device)
        reward     = batch['reward'].to(device).float() * reward_scale
        next_state = batch['next_state'].to(device).float() / 255.0   # (B, 4, 84, 84)
        done       = batch['done'].to(device).float()

        # Ensure all tensors are 1D to prevent silent broadcasting bugs
        if reward.dim() == 2:
            reward = reward.squeeze(1)  # (B,)
        if done.dim() == 2:
            done = done.squeeze(1)      # (B,)

        # Convert one-hot action to class index if needed
        if action.dim() == 2:
            action_idx = action.argmax(dim=-1)  # (B,)
        else:
            action_idx = action                  # (B,)

        B = state.size(0)

        # ── Extract features once (CNN is frozen; no_grad is correct and efficient) ──
        with torch.no_grad():
            feat      = model.cnn(state)       # (B, 3136)
            next_feat = model.cnn(next_state)  # (B, 3136)

        # ── Compute all stable target values before any optimizer step ────────────
        with torch.no_grad():
            # V(s') for Q-loss TD target
            v_next   = model.v_network(next_feat).squeeze(1)          # (B,)
            q_target = reward + gamma * (1.0 - done) * v_next         # (B,)

            # Q_target(s, a_data): stable reference used in V loss and Actor loss
            # Using target Q prevents the Q-network from chasing its own bootstrap
            q_t_all = model.q_network_target(feat)                    # (B, action_dim)
            q_t_a   = q_t_all.gather(                                 # (B,)
                1, action_idx.unsqueeze(1)
            ).squeeze(1)

        # ─────────────────────────────────────────────────────────────────────────
        # Step 1 — V loss: expectile regression
        #
        # V(s) should sit at the tau-th quantile of Q over the data distribution.
        # diff = Q_target(s, a_data) - V(s)
        # L_V = mean(tau * diff^2  if diff > 0
        #            (1-tau) * diff^2  otherwise)
        #
        # tau > 0.5 biases V to be an optimistic (upper-quantile) value estimate,
        # which is key to IQL's safe offline bootstrap.
        # ─────────────────────────────────────────────────────────────────────────
        v_pred = model.v_network(feat).squeeze(1)                     # (B,)
        diff   = q_t_a - v_pred                                       # (B,)
        ew     = torch.where(
            diff > 0,
            torch.full_like(diff, model.tau),
            torch.full_like(diff, 1.0 - model.tau)
        )                                                             # (B,)
        v_loss = (ew * diff.pow(2)).mean()                            # scalar

        v_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.v_network.parameters(), 10.0)
        v_optimizer.step()

        # ─────────────────────────────────────────────────────────────────────────
        # Step 2 — Q loss: TD regression
        #
        # Q(s, a_data) → r + γ * (1 - done) * V(s')
        # q_target was computed with no_grad before any optimizer step, so
        # V's update in Step 1 does not affect this batch's Q target.
        # ─────────────────────────────────────────────────────────────────────────
        q_pred_all = model.q_network(feat)                            # (B, action_dim)
        q_pred     = q_pred_all.gather(                               # (B,)
            1, action_idx.unsqueeze(1)
        ).squeeze(1)
        q_loss = F.smooth_l1_loss(q_pred, q_target)                  # scalar

        q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.q_network.parameters(), 10.0)
        q_optimizer.step()

        # ─────────────────────────────────────────────────────────────────────────
        # Step 3 — Actor loss: advantage-weighted behavior cloning
        #
        # adv = Q_target(s, a_data) - V(s)    [both detached from actor graph]
        # w   = clamp(exp(β * adv), 0, max_weight)
        # L_actor = -mean(w * log π(a_data | s))
        #
        # Using Q_target instead of online Q prevents the actor from chasing
        # a moving Q estimate within the same update step.
        # Using the freshly-stepped V (post Step-1) gives slightly more accurate
        # advantage estimates than the pre-update V.
        # ─────────────────────────────────────────────────────────────────────────
        with torch.no_grad():
            v_curr = model.v_network(feat).squeeze(1)                 # (B,) — post Step-1
            adv    = q_t_a - v_curr                                   # (B,)

            # Clamp β*adv inside exp to prevent numerical overflow before exponentiation
            # log(max_weight) is the maximum value we allow β*adv to reach
            w = torch.exp(
                torch.clamp(model.beta * adv, max=math.log(model.max_weight))
            ).clamp(max=model.max_weight)                             # (B,)

        actor_logits = model.actor(feat)                              # (B, action_dim)
        log_probs    = F.log_softmax(actor_logits, dim=-1)            # (B, action_dim)
        log_prob_a   = log_probs.gather(                              # (B,)
            1, action_idx.unsqueeze(1)
        ).squeeze(1)
        actor_loss   = -(w * log_prob_a).mean()                       # scalar

        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 10.0)
        actor_optimizer.step()

        # ── Periodic hard target Q update ─────────────────────────────────────────
        if model.training_step % target_update_freq == 0:
            model.update_target()

        # Statistics
        total_q_loss     += q_loss.item()     * B
        total_v_loss     += v_loss.item()     * B
        total_actor_loss += actor_loss.item() * B
        total_q_value    += q_pred_all.mean().item() * B
        total_v_value    += v_curr.mean().item()     * B
        total_adv        += adv.mean().item()        * B
        total_weight     += w.mean().item()          * B
        total_samples    += B

    avg_q_loss     = total_q_loss     / total_samples
    avg_v_loss     = total_v_loss     / total_samples
    avg_actor_loss = total_actor_loss / total_samples
    avg_q_value    = total_q_value    / total_samples
    avg_v_value    = total_v_value    / total_samples
    avg_adv        = total_adv        / total_samples
    avg_weight     = total_weight     / total_samples

    return avg_q_loss, avg_v_loss, avg_actor_loss, avg_q_value, avg_v_value, avg_adv, avg_weight


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def val_iql(model, dataloader, device, gamma=0.99, reward_scale=0.1, action_weights=None):
    """Validation function for IQL with detailed metrics.

    Returns:
        avg_q_loss, avg_v_loss, avg_actor_loss,
        avg_q_value, avg_v_value, avg_adv, avg_weight,
        avg_ce_loss, avg_wce_loss, action_accuracy
    """
    model.eval()

    total_q_loss     = 0.0
    total_v_loss     = 0.0
    total_actor_loss = 0.0
    total_q_value    = 0.0
    total_v_value    = 0.0
    total_adv        = 0.0
    total_weight     = 0.0
    total_ce_loss    = 0.0
    total_wce_loss   = 0.0
    total_correct    = 0
    total_samples    = 0

    with torch.no_grad():
        for batch in dataloader:
            state      = batch['state'].to(device).float() / 255.0
            action     = batch['action'].to(device)
            reward     = batch['reward'].to(device).float() * reward_scale
            next_state = batch['next_state'].to(device).float() / 255.0
            done       = batch['done'].to(device).float()

            # Ensure 1D to prevent broadcasting bugs
            if reward.dim() == 2:
                reward = reward.squeeze(1)  # (B,)
            if done.dim() == 2:
                done = done.squeeze(1)      # (B,)

            # Convert one-hot action to class index
            if action.dim() == 2:
                action_idx = action.argmax(dim=-1)  # (B,)
            else:
                action_idx = action                  # (B,)

            B = state.size(0)

            # Extract features (CNN frozen)
            feat      = model.cnn(state)       # (B, 3136)
            next_feat = model.cnn(next_state)  # (B, 3136)

            # ── Q loss ──────────────────────────────────────────────────────
            v_next     = model.v_network(next_feat).squeeze(1)         # (B,)
            q_target   = reward + gamma * (1.0 - done) * v_next        # (B,)

            q_pred_all = model.q_network(feat)                         # (B, action_dim)
            q_pred     = q_pred_all.gather(                            # (B,)
                1, action_idx.unsqueeze(1)
            ).squeeze(1)
            q_loss     = F.smooth_l1_loss(q_pred, q_target)

            # ── V loss ──────────────────────────────────────────────────────
            q_t_all = model.q_network_target(feat)                     # (B, action_dim)
            q_t_a   = q_t_all.gather(                                  # (B,)
                1, action_idx.unsqueeze(1)
            ).squeeze(1)

            v_pred = model.v_network(feat).squeeze(1)                  # (B,)
            diff   = q_t_a - v_pred                                    # (B,)
            ew     = torch.where(
                diff > 0,
                torch.full_like(diff, model.tau),
                torch.full_like(diff, 1.0 - model.tau)
            )                                                          # (B,)
            v_loss = (ew * diff.pow(2)).mean()

            # ── Advantages and weights ───────────────────────────────────────
            adv = q_t_a - v_pred                                       # (B,)
            w   = torch.exp(
                torch.clamp(model.beta * adv, max=math.log(model.max_weight))
            ).clamp(max=model.max_weight)                              # (B,)

            # ── Actor loss ──────────────────────────────────────────────────
            actor_logits = model.actor(feat)                           # (B, action_dim)
            log_probs    = F.log_softmax(actor_logits, dim=-1)         # (B, action_dim)
            log_prob_a   = log_probs.gather(                           # (B,)
                1, action_idx.unsqueeze(1)
            ).squeeze(1)
            actor_loss   = -(w * log_prob_a).mean()

            # ── Action accuracy (greedy actor vs dataset action) ─────────────
            action_pred = actor_logits.argmax(dim=-1)                  # (B,)

            # ── CE / wCE on Q-values for cross-method comparison ─────────────
            # Z-score normalize Q-values to use as logits (same convention as val_cql)
            q_mean   = q_pred_all.mean(dim=-1, keepdim=True)
            q_std    = q_pred_all.std(dim=-1, keepdim=True) + 1e-8
            z_values = (q_pred_all - q_mean) / q_std                  # (B, action_dim)
            ce_loss  = F.cross_entropy(z_values, action_idx)
            wce_loss = F.cross_entropy(z_values, action_idx, weight=action_weights)

            # Statistics
            total_q_loss     += q_loss.item()     * B
            total_v_loss     += v_loss.item()     * B
            total_actor_loss += actor_loss.item() * B
            total_q_value    += q_pred_all.mean().item() * B
            total_v_value    += v_pred.mean().item()     * B
            total_adv        += adv.mean().item()        * B
            total_weight     += w.mean().item()          * B
            total_ce_loss    += ce_loss.item()           * B
            total_wce_loss   += wce_loss.item()          * B
            total_correct    += (action_pred == action_idx).sum().item()
            total_samples    += B

    avg_q_loss     = total_q_loss     / total_samples
    avg_v_loss     = total_v_loss     / total_samples
    avg_actor_loss = total_actor_loss / total_samples
    avg_q_value    = total_q_value    / total_samples
    avg_v_value    = total_v_value    / total_samples
    avg_adv        = total_adv        / total_samples
    avg_weight     = total_weight     / total_samples
    avg_ce_loss    = total_ce_loss    / total_samples
    avg_wce_loss   = total_wce_loss   / total_samples
    action_accuracy = total_correct   / total_samples

    return (avg_q_loss, avg_v_loss, avg_actor_loss,
            avg_q_value, avg_v_value, avg_adv, avg_weight,
            avg_ce_loss, avg_wce_loss, action_accuracy)