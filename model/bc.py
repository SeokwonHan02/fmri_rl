import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .dqn import DQN_CNN

class BehaviorCloning(nn.Module):
    """
    Behavior Cloning model
    Architecture: DQN_CNN -> MLP(3136 -> 512 -> 6)
    """
    def __init__(self, cnn, hidden_dim=512, action_dim=6):
        super(BehaviorCloning, self).__init__()

        self.cnn = cnn
        self.action_dim = action_dim

        # MLP layers (randomly initialized)
        self.mlp = nn.Sequential(
            nn.Linear(3136, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        features = self.cnn(state)  # (batch, 3136)
        action_logits = self.mlp(features)  # (batch, 6)
        return action_logits

    def get_action(self, state, deterministic=True):
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(state)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)

        return action.item() if action.numel() == 1 else action


def train_bc(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training BC", leave=False)
    for batch in pbar:
        state = batch['state'].to(device).float() / 255.0  # Normalize to [0, 1]
        action = batch['action'].to(device)

        # Convert one-hot action to class index
        if action.dim() == 2:
            action = action.argmax(dim=-1)

        # Forward pass
        logits = model(state)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, action)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Statistics
        total_loss += loss.item() * state.size(0)
        pred = logits.argmax(dim=-1)
        total_correct += (pred == action).sum().item()
        total_samples += state.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{(pred == action).float().mean().item():.4f}'
        })

    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy
