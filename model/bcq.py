import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .dqn import DQN_CNN
import copy

class BCQ(nn.Module):
    """
    Batch Constrained Q-learning (BCQ) for discrete actions
    Architecture: DQN_CNN (frozen, outputs 3136) -> Linear(3136 -> 512) -> Linear(512 -> 6)

    Consists of:
    1. Q-network: Estimates Q-values (3136 -> 512 -> 6)
    2. Imitation network: Learns behavior policy (3136 -> 512 -> 6)
    3. Target Q-network: For stable Q-learning
    """
    def __init__(self, cnn, action_dim=6, threshold=0.3, logit_div=1.0, bc_path=''):
        super(BCQ, self).__init__()

        self.action_dim = action_dim
        self.threshold = threshold
        self.logit_div = logit_div
        self.bc_frozen = (bc_path != '')  # Track if BC network is frozen

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

        # Imitation network: 3136 -> 512 -> 6
        self.imitation_network = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        # Load pretrained BC model if provided
        if bc_path:
            print(f"Loading pretrained BC model from: {bc_path}")
            from .bc import BehaviorCloning

            # Load pretrained BC model
            bc_state_dict = torch.load(bc_path, map_location='cpu')

            # Create temporary BC model to extract action_head weights
            temp_bc = BehaviorCloning(cnn, action_dim=action_dim, logit_div=logit_div)
            temp_bc.load_state_dict(bc_state_dict)

            # Copy weights from BC's action_head to imitation_network
            self.imitation_network.load_state_dict(temp_bc.action_head.state_dict())

            # Freeze imitation network
            for param in self.imitation_network.parameters():
                param.requires_grad = False

            print("BC network loaded and frozen")

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
        imitation_logits = self.imitation_network(features)
        return q_values, imitation_logits

    def get_action(self, state, deterministic=True):
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            q_values, imitation_logits = self.forward(state)

            # Apply temperature scaling to imitation logits
            scaled_logits = imitation_logits / self.logit_div

            # Get imitation probabilities
            imitation_probs = F.softmax(scaled_logits, dim=-1)

            # Mask out actions with low imitation probability
            # Only consider actions where prob > (max_prob * threshold)
            max_prob = imitation_probs.max(dim=-1, keepdim=True)[0]
            mask = imitation_probs > (max_prob * self.threshold)

            # Set Q-values of masked actions to -inf
            masked_q = q_values.clone()
            masked_q[~mask] = -float('inf')

            # Select action with highest Q-value
            action = masked_q.argmax(dim=-1)

        return action.item() if action.numel() == 1 else action

    def update_target(self):
        self.q_network_target.load_state_dict(self.q_network.state_dict())


def train_bcq(model, dataloader, optimizer, device, gamma=0.99, target_update_freq=1000, label_smoothing=0.0):
    model.train()
    total_q_loss = 0
    total_bc_loss = 0
    total_bc_correct = 0
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
        q_values, imitation_logits = model(state)
        q_value = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # (Batch,)

        # Compute target Q-value
        with torch.no_grad():
            # NOTE: Using online CNN here. Safe because CNN is frozen in __init__
            next_features = model.cnn(next_state)
            next_q_values = model.q_network_target(next_features)

            # Get next imitation probabilities (from current imitation network)
            next_imitation_logits = model.imitation_network(next_features)
            next_imitation_probs = F.softmax(next_imitation_logits, dim=-1)

            # Mask out actions with low imitation probability
            # Only consider actions where prob > (max_prob * threshold)
            max_prob = next_imitation_probs.max(dim=-1, keepdim=True)[0]
            mask = next_imitation_probs > (max_prob * model.threshold)
            if not mask.any(dim=-1).all():
                mask = torch.ones_like(mask, dtype=torch.bool)

            masked_next_q = next_q_values.clone()
            masked_next_q[~mask] = -float('inf')

            # Max Q-value among valid actions
            next_q_value = masked_next_q.max(dim=1)[0]  # (Batch,)
            # Now all are (Batch,): reward, done, next_q_value
            target_q = reward + gamma * next_q_value * (1 - done)

        # Q-learning loss (Huber loss for stability)
        q_loss = F.smooth_l1_loss(q_value, target_q)

        # Behavior cloning loss with label smoothing (only if BC is not frozen)
        if model.bc_frozen:
            bc_loss = torch.tensor(0.0, device=q_loss.device)
            total_loss = q_loss
        else:
            bc_loss = F.cross_entropy(imitation_logits, action_idx, label_smoothing=label_smoothing)
            total_loss = q_loss + bc_loss

        # Single backward and optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        # Update target network
        if model.training_step % target_update_freq == 0:
            model.update_target()

        # Statistics
        total_q_loss += q_loss.item() * state.size(0)
        total_bc_loss += bc_loss.item() * state.size(0)
        pred = imitation_logits.argmax(dim=-1)
        total_bc_correct += (pred == action_idx).sum().item()
        total_q_value += q_values.mean().item() * state.size(0)
        total_samples += state.size(0)

    avg_q_loss = total_q_loss / total_samples
    avg_bc_loss = total_bc_loss / total_samples
    avg_bc_accuracy = total_bc_correct / total_samples
    avg_q_value = total_q_value / total_samples

    return avg_q_loss, avg_bc_loss, avg_bc_accuracy, avg_q_value


def val_bcq(model, dataloader, device, gamma=0.99, label_smoothing=0.0):
    """Validation function for BCQ with detailed metrics"""
    model.eval()
    total_q_loss = 0
    total_bc_loss = 0
    total_bc_correct = 0
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
            q_values, imitation_logits = model(state)
            q_value = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # (Batch,)

            # Compute target Q-value
            next_features = model.cnn(next_state)
            next_q_values = model.q_network_target(next_features)

            # Get next imitation probabilities
            next_imitation_logits = model.imitation_network(next_features)
            next_imitation_probs = F.softmax(next_imitation_logits, dim=-1)

            # Mask out actions with low imitation probability
            max_prob = next_imitation_probs.max(dim=-1, keepdim=True)[0]
            mask = next_imitation_probs > (max_prob * model.threshold)
            if not mask.any(dim=-1).all():
                mask = torch.ones_like(mask, dtype=torch.bool)

            masked_next_q = next_q_values.clone()
            masked_next_q[~mask] = -float('inf')

            # Max Q-value among valid actions
            next_q_value = masked_next_q.max(dim=1)[0]  # (Batch,)
            target_q = reward + gamma * next_q_value * (1 - done)

            # Q-learning loss
            q_loss = F.smooth_l1_loss(q_value, target_q)

            # Behavior cloning loss with label smoothing (only if BC is not frozen)
            if model.bc_frozen:
                bc_loss = torch.tensor(0.0, device=q_loss.device)
            else:
                bc_loss = F.cross_entropy(imitation_logits, action_idx, label_smoothing=label_smoothing)

            # Statistics
            total_q_loss += q_loss.item() * state.size(0)
            total_bc_loss += bc_loss.item() * state.size(0)
            pred = imitation_logits.argmax(dim=-1)
            total_bc_correct += (pred == action_idx).sum().item()
            total_q_value += q_values.mean().item() * state.size(0)
            total_samples += state.size(0)

    avg_q_loss = total_q_loss / total_samples
    avg_bc_loss = total_bc_loss / total_samples
    avg_bc_accuracy = total_bc_correct / total_samples
    avg_q_value = total_q_value / total_samples

    return avg_q_loss, avg_bc_loss, avg_bc_accuracy, avg_q_value
