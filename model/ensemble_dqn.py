import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class EnsembleDQN(nn.Module):
    """
    Bootstrapped DQN with ensemble of independent MLP heads.
    Architecture: DQN_CNN (frozen, outputs 3136) -> num_heads x [Linear(3136->512)->ReLU->Linear(512->6)]

    The encoder is always frozen; only the MLP heads are trained.
    Each head has a corresponding target head for stable Q-learning.
    """
    def __init__(self, cnn, num_heads=20, action_dim=6, hidden_dim=512):
        super(EnsembleDQN, self).__init__()

        self.action_dim = action_dim
        self.num_heads = num_heads

        # CNN (always frozen)
        self.cnn = cnn
        for param in self.cnn.parameters():
            param.requires_grad = False
        print("✓ EnsembleDQN CNN: All conv layers frozen")

        # Ensemble of independent MLP heads: num_heads x [3136 -> 512 -> action_dim]
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3136, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            for _ in range(num_heads)
        ])
        print(f"✓ EnsembleDQN: {num_heads} MLP heads initialized")

        # Target heads (frozen copies, updated periodically via update_target())
        self.target_heads = nn.ModuleList([
            copy.deepcopy(head) for head in self.heads
        ])
        for head in self.target_heads:
            for param in head.parameters():
                param.requires_grad = False

        # Training step counter (for target network update scheduling)
        self.training_step = 0

    def forward(self, state):
        """
        Args:
            state: (batch, 4, 84, 84)
        Returns:
            q_all: (batch, num_heads, action_dim)
        """
        features = self.cnn(state)  # (batch, 3136), frozen via requires_grad=False
        q_all = torch.stack([head(features) for head in self.heads], dim=1)
        return q_all

    def get_action(self, state, deterministic=True):
        """Select action using mean Q-values across all heads."""
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            q_all = self.forward(state)
            mean_q = q_all.mean(dim=1)
            action = mean_q.argmax(dim=-1)

        return action.item() if action.numel() == 1 else action

    def update_target(self):
        """Hard update: copy current head weights to target heads."""
        for head, target_head in zip(self.heads, self.target_heads):
            target_head.load_state_dict(head.state_dict())


def train_ensemble_dqn(model, dataloader, optimizer, device, gamma=0.99,
                        target_update_freq=100, reward_scale=0.1, mask_prob=1.0,
                        ce_lambda=1.0):
    """
    Train bootstrapped ensemble DQN with Double DQN targets + BC regularization.

    Each head uses its own online Q to select the next action (Double DQN),
    then evaluates that action with its own target Q.
    Bernoulli masks provide bootstrap diversity across heads.
    CE loss on z-score normalized mean Q regularizes toward the data distribution.

    total_loss = td_loss + ce_lambda * ce_loss

    Returns:
        avg_td_loss, avg_ce_loss, avg_total_loss, avg_q_value
    """
    model.train()
    total_td_loss    = 0
    total_ce_loss    = 0
    total_total_loss = 0
    total_q_value    = 0
    total_samples    = 0

    for batch in dataloader:
        model.training_step += 1
        state      = batch['state'].to(device).float() / 255.0
        action     = batch['action'].to(device)
        reward     = batch['reward'].to(device).float() * reward_scale
        next_state = batch['next_state'].to(device).float() / 255.0
        done       = batch['done'].to(device).float()

        if reward.dim() == 2:
            reward = reward.squeeze(1)
        if done.dim() == 2:
            done = done.squeeze(1)

        if action.dim() == 2:
            action_idx = action.argmax(dim=-1)
        else:
            action_idx = action.long()

        batch_size = state.size(0)

        # Bootstrapped masks: each (sample, head) pair included with prob mask_prob
        masks = torch.bernoulli(
            torch.full((batch_size, model.num_heads), mask_prob, device=device)
        )  # (batch_size, num_heads)

        # Current Q-values from all heads (with gradients for head parameters)
        current_q_all = model(state)  # (batch_size, num_heads, action_dim)

        # Double DQN target: online heads select action, target heads evaluate it
        with torch.no_grad():
            next_features = model.cnn(next_state)  # (batch_size, 3136)

            # Online heads select best next action per head
            next_q_online = torch.stack(
                [head(next_features) for head in model.heads], dim=1
            )  # (batch_size, num_heads, action_dim)
            next_best_actions = next_q_online.argmax(dim=-1, keepdim=True)  # (batch_size, num_heads, 1)

            # Target heads evaluate those actions
            next_q_targets = torch.stack(
                [head(next_features) for head in model.target_heads], dim=1
            )  # (batch_size, num_heads, action_dim)
            next_q_value = next_q_targets.gather(2, next_best_actions).squeeze(2)  # (batch_size, num_heads)

            target_q = (reward.unsqueeze(1) +
                        gamma * next_q_value * (1 - done.unsqueeze(1)))  # (batch_size, num_heads)

        # Gather Q-values for taken actions across all heads
        action_indices = action_idx.view(-1, 1, 1).expand(-1, model.num_heads, 1)
        current_q = current_q_all.gather(2, action_indices).squeeze(2)  # (batch_size, num_heads)

        # Masked TD loss: each head only updates on its bootstrap sample
        td_loss_per = F.smooth_l1_loss(current_q, target_q, reduction='none')  # (batch_size, num_heads)
        mask_sum = masks.sum().clamp_min(1.0)
        td_loss = (td_loss_per * masks).sum() / mask_sum

        # BC regularization: z-score normalize mean Q, compute CE loss toward data actions
        mean_q  = current_q_all.mean(dim=1)                            # (batch_size, action_dim)
        q_mean  = mean_q.mean(dim=-1, keepdim=True)
        q_std   = mean_q.std(dim=-1, keepdim=True) + 1e-8
        z_values = (mean_q - q_mean) / q_std
        ce_loss = F.cross_entropy(z_values, action_idx)

        total_loss = td_loss + ce_lambda * ce_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.heads.parameters(), 10.0)
        optimizer.step()

        # Periodic target network update
        if model.training_step % target_update_freq == 0:
            model.update_target()

        total_td_loss    += td_loss.item()    * batch_size
        total_ce_loss    += ce_loss.item()    * batch_size
        total_total_loss += total_loss.item() * batch_size
        total_q_value    += mean_q.detach().mean().item() * batch_size
        total_samples    += batch_size

    return (
        total_td_loss    / total_samples,
        total_ce_loss    / total_samples,
        total_total_loss / total_samples,
        total_q_value    / total_samples,
    )


def val_ensemble_dqn(model, dataloader, device, gamma=0.99, reward_scale=0.1,
                     action_weights=None):
    """
    Validation for EnsembleDQN with Double DQN targets.

    Uses mean Q-values across heads for TD loss and CE metrics.
    Online mean Q selects next actions; target mean Q evaluates them.

    Returns:
        avg_td_loss, avg_q_value, avg_ce_loss, avg_wce_loss, action_accuracy
    """
    model.eval()
    total_td_loss  = 0
    total_q_value  = 0
    total_ce_loss  = 0
    total_wce_loss = 0
    total_correct  = 0
    total_samples  = 0

    with torch.no_grad():
        for batch in dataloader:
            state      = batch['state'].to(device).float() / 255.0
            action     = batch['action'].to(device)
            reward     = batch['reward'].to(device).float() * reward_scale
            next_state = batch['next_state'].to(device).float() / 255.0
            done       = batch['done'].to(device).float()

            if reward.dim() == 2:
                reward = reward.squeeze(1)
            if done.dim() == 2:
                done = done.squeeze(1)

            if action.dim() == 2:
                action_idx = action.argmax(dim=-1)
            else:
                action_idx = action.long()

            batch_size = state.size(0)

            # Compute features once (CNN frozen)
            features      = model.cnn(state)
            next_features = model.cnn(next_state)

            # Current Q-values from all heads
            current_q_all = torch.stack(
                [head(features) for head in model.heads], dim=1
            )  # (batch_size, num_heads, action_dim)
            mean_q = current_q_all.mean(dim=1)  # (batch_size, action_dim)

            # Double DQN: online mean Q selects action, target mean Q evaluates it
            next_q_online = torch.stack(
                [head(next_features) for head in model.heads], dim=1
            )  # (batch_size, num_heads, action_dim)
            next_best_actions = next_q_online.mean(dim=1).argmax(dim=-1, keepdim=True)  # (batch_size, 1)

            next_q_targets = torch.stack(
                [head(next_features) for head in model.target_heads], dim=1
            )  # (batch_size, num_heads, action_dim)
            next_q_value = next_q_targets.mean(dim=1).gather(1, next_best_actions).squeeze(1)  # (batch_size,)
            target_q = reward + gamma * next_q_value * (1 - done)

            # TD loss using mean Q-values
            q_value = mean_q.gather(1, action_idx.unsqueeze(1)).squeeze(1)
            td_loss = F.smooth_l1_loss(q_value, target_q)

            # CE loss: Z-score normalize mean Q-values
            q_mean   = mean_q.mean(dim=-1, keepdim=True)
            q_std    = mean_q.std(dim=-1, keepdim=True) + 1e-8
            z_values = (mean_q - q_mean) / q_std
            ce_loss  = F.cross_entropy(z_values, action_idx)
            wce_loss = F.cross_entropy(z_values, action_idx, weight=action_weights)
            action_pred = z_values.argmax(dim=-1)

            total_td_loss  += td_loss.item() * batch_size
            total_q_value  += mean_q.mean().item() * batch_size
            total_ce_loss  += ce_loss.item() * batch_size
            total_wce_loss += wce_loss.item() * batch_size
            total_correct  += (action_pred == action_idx).sum().item()
            total_samples  += batch_size

    return (
        total_td_loss  / total_samples,
        total_q_value  / total_samples,
        total_ce_loss  / total_samples,
        total_wce_loss / total_samples,
        total_correct  / total_samples,
    )
