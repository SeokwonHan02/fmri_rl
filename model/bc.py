import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .dqn import DQN_CNN

class BehaviorCloning(nn.Module):
    """
    Behavior Cloning model with factorized action space
    Architecture: DQN_CNN -> Shared FC -> Fire Head (2-class) + Move Head (3-class)

    Fire: Not Fire (NOOP, RIGHT, LEFT) vs Yes Fire (FIRE, RIGHT+FIRE, LEFT+FIRE)
    Move: Not Move (NOOP, FIRE) vs Right (RIGHT, RIGHT+FIRE) vs Left (LEFT, LEFT+FIRE)
    """
    def __init__(self, cnn, action_dim=6, logit_div=1.0):
        super(BehaviorCloning, self).__init__()

        self.cnn = cnn
        self.action_dim = action_dim
        self.logit_div = logit_div

        # Shared feature layer: 3136 -> 512
        self.shared_fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        # Fire head: binary classification (not fire vs yes fire)
        self.fire_head = nn.Linear(512, 2)

        # Move head: 3-class classification (not move, right, left)
        self.move_head = nn.Linear(512, 3)

    def forward(self, state):
        features = self.cnn(state)  # (batch, 3136)
        shared = self.shared_fc(features)  # (batch, 512)

        fire_logits = self.fire_head(shared)  # (batch, 2)
        move_logits = self.move_head(shared)  # (batch, 3)

        return fire_logits, move_logits

    def get_action(self, state, deterministic=True):
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            fire_logits, move_logits = self.forward(state)

            # Apply temperature scaling
            fire_logits = fire_logits / self.logit_div
            move_logits = move_logits / self.logit_div

            if deterministic:
                fire_pred = fire_logits.argmax(dim=-1)  # 0 or 1
                move_pred = move_logits.argmax(dim=-1)  # 0, 1, or 2
            else:
                fire_probs = F.softmax(fire_logits, dim=-1)
                move_probs = F.softmax(move_logits, dim=-1)
                fire_pred = torch.multinomial(fire_probs, 1).squeeze(-1)
                move_pred = torch.multinomial(move_probs, 1).squeeze(-1)

            # Reconstruct action from fire and move predictions
            action = self._reconstruct_action(fire_pred, move_pred)

        return action.item() if action.numel() == 1 else action

    def _reconstruct_action(self, fire_pred, move_pred):
        """
        Reconstruct original action from fire and move predictions

        Mapping:
        - (fire=0, move=0) -> NOOP (0)
        - (fire=1, move=0) -> FIRE (1)
        - (fire=0, move=1) -> RIGHT (2)
        - (fire=0, move=2) -> LEFT (3)
        - (fire=1, move=1) -> RIGHT+FIRE (4)
        - (fire=1, move=2) -> LEFT+FIRE (5)
        """
        # Create mapping tensor: [fire][move] -> action
        action_map = torch.tensor([
            [0, 2, 3],  # fire=0: NOOP, RIGHT, LEFT
            [1, 4, 5],  # fire=1: FIRE, RIGHT+FIRE, LEFT+FIRE
        ], device=fire_pred.device)

        # Index into the mapping
        action = action_map[fire_pred, move_pred]
        return action


def action_to_fire_move(action):
    """
    Convert action index to fire and move labels

    Action mapping:
    - 0: NOOP       -> fire=0, move=0
    - 1: FIRE       -> fire=1, move=0
    - 2: RIGHT      -> fire=0, move=1
    - 3: LEFT       -> fire=0, move=2
    - 4: RIGHT+FIRE -> fire=1, move=1
    - 5: LEFT+FIRE  -> fire=1, move=2

    Args:
        action: (batch,) tensor of action indices (0-5)

    Returns:
        fire_label: (batch,) tensor of fire labels (0 or 1)
        move_label: (batch,) tensor of move labels (0, 1, or 2)
    """
    # Fire mapping: 0->0, 1->1, 2->0, 3->0, 4->1, 5->1
    fire_map = torch.tensor([0, 1, 0, 0, 1, 1], dtype=torch.long, device=action.device)
    fire_label = fire_map[action]

    # Move mapping: 0->0, 1->0, 2->1, 3->2, 4->1, 5->2
    move_map = torch.tensor([0, 0, 1, 2, 1, 2], dtype=torch.long, device=action.device)
    move_label = move_map[action]

    return fire_label, move_label


def train_bc(model, dataloader, optimizer, device, label_smoothing=0.0):
    model.train()
    total_fire_loss = 0
    total_move_loss = 0
    total_loss = 0
    total_fire_correct = 0
    total_move_correct = 0
    total_action_correct = 0
    total_samples = 0

    for batch in dataloader:
        state = batch['state'].to(device).float() / 255.0  # Normalize to [0, 1]
        action = batch['action'].to(device)

        # Convert one-hot action to class index
        if action.dim() == 2:
            action = action.argmax(dim=-1)

        # Convert action to fire and move labels
        fire_label, move_label = action_to_fire_move(action)
        fire_label = fire_label.to(device)
        move_label = move_label.to(device)

        # Forward pass
        fire_logits, move_logits = model(state)

        # Compute losses (fire binary CE + move multi-class CE)
        fire_loss = F.cross_entropy(fire_logits, fire_label, label_smoothing=label_smoothing)
        move_loss = F.cross_entropy(move_logits, move_label, label_smoothing=label_smoothing)
        loss = fire_loss + move_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        # Statistics
        fire_pred = fire_logits.argmax(dim=-1)
        move_pred = move_logits.argmax(dim=-1)
        action_pred = model._reconstruct_action(fire_pred, move_pred)

        total_fire_loss += fire_loss.item() * state.size(0)
        total_move_loss += move_loss.item() * state.size(0)
        total_loss += loss.item() * state.size(0)
        total_fire_correct += (fire_pred == fire_label).sum().item()
        total_move_correct += (move_pred == move_label).sum().item()
        total_action_correct += (action_pred == action).sum().item()
        total_samples += state.size(0)

    avg_fire_loss = total_fire_loss / total_samples
    avg_move_loss = total_move_loss / total_samples
    avg_loss = total_loss / total_samples
    fire_accuracy = total_fire_correct / total_samples
    move_accuracy = total_move_correct / total_samples
    action_accuracy = total_action_correct / total_samples

    return avg_loss, action_accuracy, avg_fire_loss, avg_move_loss, fire_accuracy, move_accuracy


def val_bc(model, dataloader, device, label_smoothing=0.0):
    """Validation function for Behavior Cloning"""
    model.eval()
    total_fire_loss = 0
    total_move_loss = 0
    total_loss = 0
    total_fire_correct = 0
    total_move_correct = 0
    total_action_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device).float() / 255.0
            action = batch['action'].to(device)

            # Convert one-hot action to class index
            if action.dim() == 2:
                action = action.argmax(dim=-1)

            # Convert action to fire and move labels
            fire_label, move_label = action_to_fire_move(action)
            fire_label = fire_label.to(device)
            move_label = move_label.to(device)

            # Forward pass
            fire_logits, move_logits = model(state)

            # Compute losses
            fire_loss = F.cross_entropy(fire_logits, fire_label, label_smoothing=label_smoothing)
            move_loss = F.cross_entropy(move_logits, move_label, label_smoothing=label_smoothing)
            loss = fire_loss + move_loss

            # Statistics
            fire_pred = fire_logits.argmax(dim=-1)
            move_pred = move_logits.argmax(dim=-1)
            action_pred = model._reconstruct_action(fire_pred, move_pred)

            total_fire_loss += fire_loss.item() * state.size(0)
            total_move_loss += move_loss.item() * state.size(0)
            total_loss += loss.item() * state.size(0)
            total_fire_correct += (fire_pred == fire_label).sum().item()
            total_move_correct += (move_pred == move_label).sum().item()
            total_action_correct += (action_pred == action).sum().item()
            total_samples += state.size(0)

    avg_fire_loss = total_fire_loss / total_samples
    avg_move_loss = total_move_loss / total_samples
    avg_loss = total_loss / total_samples
    fire_accuracy = total_fire_correct / total_samples
    move_accuracy = total_move_correct / total_samples
    action_accuracy = total_action_correct / total_samples

    return avg_loss, action_accuracy, avg_fire_loss, avg_move_loss, fire_accuracy, move_accuracy
