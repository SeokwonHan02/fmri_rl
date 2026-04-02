import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class CQL(nn.Module):
    """
    Conservative Q-Learning (CQL) for discrete actions
    Architecture: DQN_CNN (frozen, outputs 3136) -> Linear(3136 -> 512) -> Linear(512 -> 6)

    Adds a conservative regularization term to standard Q-learning
    to prevent overestimation of out-of-distribution actions
    """
    def __init__(self, cnn, action_dim=6, alpha=0.2):
        super(CQL, self).__init__()

        self.action_dim = action_dim
        self.alpha = alpha  # CQL regularization weight

        # CNN (always frozen for stability)
        self.cnn = cnn
        for param in self.cnn.parameters():
            param.requires_grad = False
        print("✓ CQL CNN: All conv layers frozen")

        # Q-network: 3136 -> 512 -> 6 (randomly initialized)
        self.q_network = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        # Target Q-network (CNN is frozen)
        self.q_network_target = copy.deepcopy(self.q_network)

        # Freeze target network
        for param in self.q_network_target.parameters():
            param.requires_grad = False

        # Training step counter (for target network updates)
        self.training_step = 0

    def forward(self, state):
        features = self.cnn(state)
        q_values = self.q_network(features)
        return q_values

    def get_action(self, state, deterministic=True):
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            q_values = self.forward(state)
            action = q_values.argmax(dim=-1)

        return action.item() if action.numel() == 1 else action

    def update_target(self):
        self.q_network_target.load_state_dict(self.q_network.state_dict())


def train_cql(model, dataloader, optimizer, device, gamma=0.99, target_update_freq=100, reward_scale=0.1):
    model.train()
    total_td_loss  = 0
    total_cql_loss = 0
    total_loss_sum = 0
    total_q_value  = 0
    total_samples  = 0

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
            action_idx = action

        # Forward pass
        q_values = model(state)                                          # (B, 6)
        q_value  = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # (B,)

        # TD target
        with torch.no_grad():
            next_q_values = model.q_network_target(model.cnn(next_state))
            next_q_value  = next_q_values.max(dim=1)[0]
            target_q      = reward + gamma * next_q_value * (1 - done)

        # TD loss (Huber)
        td_loss = F.smooth_l1_loss(q_value, target_q)

        # Standard discrete CQL (Kumar et al. 2020, eq. 4):
        #   CQL = E[ logsumexp_a Q(s,a) - Q(s, a_data) ]
        cql_loss   = (torch.logsumexp(q_values, dim=1) - q_value).mean()
        total_loss = td_loss + model.alpha * cql_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if model.training_step % target_update_freq == 0:
            model.update_target()

        total_td_loss  += td_loss.item()    * state.size(0)
        total_cql_loss += cql_loss.item()   * state.size(0)
        total_loss_sum += total_loss.item() * state.size(0)
        total_q_value  += q_values.mean().item() * state.size(0)
        total_samples  += state.size(0)

    return (total_td_loss  / total_samples,
            total_cql_loss / total_samples,
            total_loss_sum / total_samples,
            total_q_value  / total_samples)


def val_cql(model, dataloader, device, gamma=0.99, reward_scale=0.1, action_weights=None):
    """Validation function for CQL with detailed metrics"""
    model.eval()
    total_td_loss  = 0
    total_cql_loss = 0
    total_loss_sum = 0
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
                action_idx = action

            # Forward pass
            q_values = model(state)                                              # (B, 6)
            q_value  = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)   # (B,)

            # TD target
            next_q_values = model.q_network_target(model.cnn(next_state))
            next_q_value  = next_q_values.max(dim=1)[0]
            target_q      = reward + gamma * next_q_value * (1 - done)

            td_loss  = F.smooth_l1_loss(q_value, target_q)

            # Standard discrete CQL
            cql_loss   = (torch.logsumexp(q_values, dim=1) - q_value).mean()
            total_loss = td_loss + model.alpha * cql_loss

            # Action prediction accuracy (z-scored Q as logits)
            q_mean      = q_values.mean(dim=-1, keepdim=True)
            q_std       = q_values.std(dim=-1, keepdim=True) + 1e-8
            z_values    = (q_values - q_mean) / q_std
            ce_loss     = F.cross_entropy(z_values, action_idx)
            wce_loss    = F.cross_entropy(z_values, action_idx, weight=action_weights)
            action_pred = z_values.argmax(dim=-1)

            total_td_loss  += td_loss.item()    * state.size(0)
            total_cql_loss += cql_loss.item()   * state.size(0)
            total_loss_sum += total_loss.item() * state.size(0)
            total_q_value  += q_values.mean().item() * state.size(0)
            total_ce_loss  += ce_loss.item()    * state.size(0)
            total_wce_loss += wce_loss.item()   * state.size(0)
            total_correct  += (action_pred == action_idx).sum().item()
            total_samples  += state.size(0)

    return (total_td_loss  / total_samples,
            total_cql_loss / total_samples,
            total_loss_sum / total_samples,
            total_q_value  / total_samples,
            total_ce_loss  / total_samples,
            total_wce_loss / total_samples,
            total_correct  / total_samples)
