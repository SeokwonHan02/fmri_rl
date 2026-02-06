import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .dqn import DQN_CNN
import copy

class BCQ(nn.Module):
    """
    Batch Constrained Q-learning (BCQ) for discrete actions

    Consists of:
    1. Q-network: Estimates Q-values
    2. Imitation network: Learns behavior policy
    3. Target Q-network: For stable Q-learning
    """
    def __init__(self, cnn, hidden_dim=512, action_dim=6, threshold=0.3):
        super(BCQ, self).__init__()

        self.action_dim = action_dim
        self.threshold = threshold

        # Frozen CNN (pretrained)
        self.cnn = cnn

        # Q-network MLP
        self.q_network = nn.Sequential(
            nn.Linear(3136, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Imitation network MLP (behavior cloning)
        self.imitation_network = nn.Sequential(
            nn.Linear(3136, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Target Q-network (MLP only, CNN is frozen)
        self.q_network_target = copy.deepcopy(self.q_network)

        # Freeze target network
        for param in self.q_network_target.parameters():
            param.requires_grad = False

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

            # Get imitation probabilities
            imitation_probs = F.softmax(imitation_logits, dim=-1)

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


def train_bcq(model, dataloader, optimizer, device, gamma=0.99, scheduler=None, step=0, target_update_freq=1000):
    model.train()
    total_q_loss = 0
    total_bc_loss = 0
    total_bc_correct = 0
    total_q_value = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training BCQ", leave=False)
    for batch in pbar:
        step += 1
        state = batch['state'].to(device).float() / 255.0
        action = batch['action'].to(device)
        reward = batch['reward'].to(device).float()
        next_state = batch['next_state'].to(device).float() / 255.0
        done = batch['done'].to(device).float()

        # Convert one-hot action to class index
        if action.dim() == 2:
            action_idx = action.argmax(dim=-1)
        else:
            action_idx = action

        # Forward pass
        q_values, imitation_logits = model(state)
        q_value = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)

        # Compute target Q-value
        with torch.no_grad():
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
            next_q_value = masked_next_q.max(dim=1)[0]
            target_q = reward + gamma * next_q_value * (1 - done)

        # Q-learning loss (Huber loss for stability)
        q_loss = F.smooth_l1_loss(q_value, target_q)

        # Behavior cloning loss
        bc_loss = F.cross_entropy(imitation_logits, action_idx)

        # Combined loss (CNN is frozen, Q and Imitation heads are separate)
        total_loss = q_loss + bc_loss

        # Single backward and optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        # Update learning rate (once per batch)
        if scheduler is not None:
            scheduler.step()

        # Update target network
        if step % target_update_freq == 0:
            model.update_target()

        # Statistics
        total_q_loss += q_loss.item() * state.size(0)
        total_bc_loss += bc_loss.item() * state.size(0)
        pred = imitation_logits.argmax(dim=-1)
        total_bc_correct += (pred == action_idx).sum().item()
        total_q_value += q_values.mean().item() * state.size(0)
        total_samples += state.size(0)

        # Update progress bar
        pbar.set_postfix({
            'q_loss': f'{q_loss.item():.4f}',
            'bc_loss': f'{bc_loss.item():.4f}',
            'bc_acc': f'{(pred == action_idx).float().mean().item():.4f}',
            'avg_q': f'{q_values.mean().item():.2f}'
        })

    avg_q_loss = total_q_loss / total_samples
    avg_bc_loss = total_bc_loss / total_samples
    avg_bc_accuracy = total_bc_correct / total_samples
    avg_q_value = total_q_value / total_samples

    return avg_q_loss, avg_bc_loss, avg_bc_accuracy, avg_q_value, step


def val_bcq(model, dataloader, device, gamma=0.99):
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
            reward = batch['reward'].to(device).float()
            next_state = batch['next_state'].to(device).float() / 255.0
            done = batch['done'].to(device).float()

            # Convert one-hot action to class index
            if action.dim() == 2:
                action_idx = action.argmax(dim=-1)
            else:
                action_idx = action

            # Forward pass
            q_values, imitation_logits = model(state)
            q_value = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1)

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
            next_q_value = masked_next_q.max(dim=1)[0]
            target_q = reward + gamma * next_q_value * (1 - done)

            # Q-learning loss
            q_loss = F.smooth_l1_loss(q_value, target_q)

            # Behavior cloning loss
            bc_loss = F.cross_entropy(imitation_logits, action_idx)

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
