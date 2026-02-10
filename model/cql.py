import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .dqn import DQN_CNN
import copy

class CQL(nn.Module):
    """
    Conservative Q-Learning (CQL) for discrete actions
    Architecture: DQN_CNN (frozen, outputs 3136) -> Linear(3136 -> 512) -> Linear(512 -> 6)

    Adds a conservative regularization term to standard Q-learning
    to prevent overestimation of out-of-distribution actions
    """
    def __init__(self, cnn, action_dim=6, alpha=1.0):
        super(CQL, self).__init__()

        self.action_dim = action_dim
        self.alpha = alpha  # CQL regularization weight

        # Frozen CNN (pretrained)
        self.cnn = cnn

        # CRITICAL: Ensure CNN is frozen to prevent target instability
        for param in self.cnn.parameters():
            param.requires_grad = False

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


def train_cql(model, dataloader, optimizer, device, gamma=0.99, target_update_freq=1000):
    model.train()
    total_td_loss = 0
    total_cql_loss = 0
    total_loss_sum = 0
    total_q_value = 0
    total_samples = 0

    for batch in dataloader:
        model.training_step += 1
        state = batch['state'].to(device).float() / 255.0
        action = batch['action'].to(device)
        reward = batch['reward'].to(device).float() / 10.0  # Reward scaling
        next_state = batch['next_state'].to(device).float() / 255.0
        done = batch['done'].to(device).float()

        # Fix dimensions: ensure all tensors are 1D to prevent broadcasting bugs
        # reward, done are usually (Batch, 1) from dataloader -> squeeze to (Batch,)
        if reward.dim() == 2:
            reward = reward.squeeze(1)
        if done.dim() == 2:
            done = done.squeeze(1)

        # Convert one-hot action to class index
        if action.dim() == 2:
            action_idx = action.argmax(dim=-1)
        else:
            action_idx = action

        # Forward pass
        q_values = model(state)
        q_value = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # (Batch,)

        # Compute target Q-value
        with torch.no_grad():
            # NOTE: Using online CNN here. Safe because CNN is frozen in __init__
            next_features = model.cnn(next_state)
            next_q_values = model.q_network_target(next_features)
            next_q_value = next_q_values.max(dim=1)[0]  # (Batch,)
            # Now all are (Batch,): reward, done, next_q_value
            target_q = reward + gamma * next_q_value * (1 - done)

        # TD loss (Huber loss for stability)
        td_loss = F.smooth_l1_loss(q_value, target_q)

        # CQL loss: penalize Q-values of all actions, boost Q-value of taken action
        # CQL term: log(sum(exp(Q(s, a)))) - Q(s, a_taken)
        logsumexp_q = torch.logsumexp(q_values, dim=1)
        cql_loss = (logsumexp_q - q_value).mean()

        # Total loss
        total_loss = td_loss + model.alpha * cql_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        # Update target network
        if model.training_step % target_update_freq == 0:
            model.update_target()

        # Statistics
        total_td_loss += td_loss.item() * state.size(0)
        total_cql_loss += cql_loss.item() * state.size(0)
        total_loss_sum += total_loss.item() * state.size(0)
        total_q_value += q_values.mean().item() * state.size(0)
        total_samples += state.size(0)

    avg_td_loss = total_td_loss / total_samples
    avg_cql_loss = total_cql_loss / total_samples
    avg_total_loss = total_loss_sum / total_samples
    avg_q_value = total_q_value / total_samples

    return avg_td_loss, avg_cql_loss, avg_total_loss, avg_q_value


def val_cql(model, dataloader, device, gamma=0.99):
    """Validation function for CQL with detailed metrics"""
    model.eval()
    total_td_loss = 0
    total_cql_loss = 0
    total_loss_sum = 0
    total_q_value = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device).float() / 255.0
            action = batch['action'].to(device)
            reward = batch['reward'].to(device).float() / 10.0  # Reward scaling
            next_state = batch['next_state'].to(device).float() / 255.0
            done = batch['done'].to(device).float()

            # Fix dimensions: ensure all tensors are 1D to prevent broadcasting bugs
            if reward.dim() == 2:
                reward = reward.squeeze(1)
            if done.dim() == 2:
                done = done.squeeze(1)

            # Convert one-hot action to class index
            if action.dim() == 2:
                action_idx = action.argmax(dim=-1)
            else:
                action_idx = action

            # Forward pass
            q_values = model(state)
            q_value = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # (Batch,)

            # Compute target Q-value
            next_features = model.cnn(next_state)
            next_q_values = model.q_network_target(next_features)
            next_q_value = next_q_values.max(dim=1)[0]  # (Batch,)
            target_q = reward + gamma * next_q_value * (1 - done)

            # TD loss
            td_loss = F.smooth_l1_loss(q_value, target_q)

            # CQL loss
            logsumexp_q = torch.logsumexp(q_values, dim=1)
            cql_loss = (logsumexp_q - q_value).mean()

            # Total loss
            total_loss = td_loss + model.alpha * cql_loss

            # Statistics
            total_td_loss += td_loss.item() * state.size(0)
            total_cql_loss += cql_loss.item() * state.size(0)
            total_loss_sum += total_loss.item() * state.size(0)
            total_q_value += q_values.mean().item() * state.size(0)
            total_samples += state.size(0)

    avg_td_loss = total_td_loss / total_samples
    avg_cql_loss = total_cql_loss / total_samples
    avg_total_loss = total_loss_sum / total_samples
    avg_q_value = total_q_value / total_samples

    return avg_td_loss, avg_cql_loss, avg_total_loss, avg_q_value
