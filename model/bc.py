import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .dqn import DQN_CNN

class BehaviorCloning(nn.Module):
    """
    Behavior Cloning model
    Architecture: DQN_CNN (frozen, outputs 3136) -> Linear(3136 -> 512) -> Linear(512 -> 6)
    """
    def __init__(self, cnn, action_dim=6, logit_div=1.0):
        super(BehaviorCloning, self).__init__()

        self.cnn = cnn
        self.action_dim = action_dim
        self.logit_div = logit_div

        # Action head: 3136 -> 512 -> 6 (randomly initialized)
        self.action_head = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, state):
        features = self.cnn(state)  # (batch, 3136)
        action_logits = self.action_head(features)  # (batch, 6)
        return action_logits

    def get_action(self, state, deterministic=True):
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(state)

            # Apply temperature scaling
            scaled_logits = logits / self.logit_div

            if deterministic:
                action = scaled_logits.argmax(dim=-1)
            else:
                probs = F.softmax(scaled_logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)

        return action.item() if action.numel() == 1 else action


def train_bc(model, dataloader, optimizer, device, label_smoothing=0.0, class_weights=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        state = batch['state'].to(device).float() / 255.0  # Normalize to [0, 1]
        action = batch['action'].to(device)

        # Convert one-hot action to class index
        if action.dim() == 2:
            action = action.argmax(dim=-1)

        # Forward pass
        logits = model(state)

        # Compute cross-entropy loss with label smoothing and class weights
        loss = F.cross_entropy(logits, action, weight=class_weights, label_smoothing=label_smoothing)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        # Statistics
        total_loss += loss.item() * state.size(0)
        pred = logits.argmax(dim=-1)
        total_correct += (pred == action).sum().item()
        total_samples += state.size(0)

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy


def val_bc(model, dataloader, device, label_smoothing=0.0, class_weights=None):
    """Validation function for Behavior Cloning"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device).float() / 255.0
            action = batch['action'].to(device)

            # Convert one-hot action to class index
            if action.dim() == 2:
                action = action.argmax(dim=-1)

            # Forward pass
            logits = model(state)

            # Compute cross-entropy loss with label smoothing and class weights
            loss = F.cross_entropy(logits, action, weight=class_weights, label_smoothing=label_smoothing)

            # Statistics
            total_loss += loss.item() * state.size(0)
            pred = logits.argmax(dim=-1)
            total_correct += (pred == action).sum().item()
            total_samples += state.size(0)

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy
