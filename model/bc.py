import torch
import torch.nn as nn
import torch.nn.functional as F

class BehaviorCloning(nn.Module):
    """
    Behavior Cloning model
    Architecture: DQN_CNN (frozen, outputs 3136) -> Linear(3136 -> 512) -> Linear(512 -> 6)

    Loss is computed by marginalizing action probabilities into fire and move:
    Fire: Not Fire (NOOP, RIGHT, LEFT) vs Yes Fire (FIRE, RIGHT+FIRE, LEFT+FIRE)
    Move: Not Move (NOOP, FIRE) vs Right (RIGHT, RIGHT+FIRE) vs Left (LEFT, LEFT+FIRE)
    """
    def __init__(self, cnn, action_dim=6):
        super(BehaviorCloning, self).__init__()

        self.cnn = cnn
        self.action_dim = action_dim

        # Action head: 3136 -> 512 -> 6
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
            action_logits = self.forward(state)

            if deterministic:
                action = action_logits.argmax(dim=-1)
            else:
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).squeeze(-1)

        return action.item() if action.numel() == 1 else action


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
    """
    # Fire mapping: 0->0, 1->1, 2->0, 3->0, 4->1, 5->1
    fire_map = torch.tensor([0, 1, 0, 0, 1, 1], dtype=torch.long, device=action.device)
    fire_label = fire_map[action]

    # Move mapping: 0->0, 1->0, 2->1, 3->2, 4->1, 5->2
    move_map = torch.tensor([0, 0, 1, 2, 1, 2], dtype=torch.long, device=action.device)
    move_label = move_map[action]

    return fire_label, move_label


def train_bc(model, dataloader, optimizer, device, label_smoothing=0.0, fire_weights=None, move_weights=None, fire_loss_weight=1.0, move_loss_weight=1.0):
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

        # Forward pass: (batch, 6)
        action_logits = model(state)

        # Marginalize action logits into fire and move logits
        # Fire: fire=0 (NOOP, RIGHT, LEFT) vs fire=1 (FIRE, RIGHT+FIRE, LEFT+FIRE)
        fire_logits = torch.stack([
            torch.logsumexp(action_logits[:, [0, 2, 3]], dim=1),  # fire=0
            torch.logsumexp(action_logits[:, [1, 4, 5]], dim=1),  # fire=1
        ], dim=1)

        # Move: move=0 (NOOP, FIRE) vs move=1 (RIGHT, RIGHT+FIRE) vs move=2 (LEFT, LEFT+FIRE)
        move_logits = torch.stack([
            torch.logsumexp(action_logits[:, [0, 1]], dim=1),  # move=0
            torch.logsumexp(action_logits[:, [2, 4]], dim=1),  # move=1
            torch.logsumexp(action_logits[:, [3, 5]], dim=1),  # move=2
        ], dim=1)

        # Compute losses with class weights and loss balance weights
        fire_loss = F.cross_entropy(fire_logits, fire_label, weight=fire_weights, label_smoothing=label_smoothing)
        move_loss = F.cross_entropy(move_logits, move_label, weight=move_weights, label_smoothing=label_smoothing)
        loss = fire_loss_weight * fire_loss + move_loss_weight * move_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        # Statistics
        action_pred = action_logits.argmax(dim=-1)
        fire_pred = fire_logits.argmax(dim=-1)
        move_pred = move_logits.argmax(dim=-1)

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


def val_bc(model, dataloader, device, label_smoothing=0.0, fire_weights=None, move_weights=None, fire_loss_weight=1.0, move_loss_weight=1.0):
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

            # Forward pass: (batch, 6)
            action_logits = model(state)

            # Marginalize action logits into fire and move logits (same as training)
            # Fire: fire=0 (NOOP, RIGHT, LEFT) vs fire=1 (FIRE, RIGHT+FIRE, LEFT+FIRE)
            fire_logits = torch.stack([
                torch.logsumexp(action_logits[:, [0, 2, 3]], dim=1),  # fire=0
                torch.logsumexp(action_logits[:, [1, 4, 5]], dim=1),  # fire=1
            ], dim=1)

            # Move: move=0 (NOOP, FIRE) vs move=1 (RIGHT, RIGHT+FIRE) vs move=2 (LEFT, LEFT+FIRE)
            move_logits = torch.stack([
                torch.logsumexp(action_logits[:, [0, 1]], dim=1),  # move=0
                torch.logsumexp(action_logits[:, [2, 4]], dim=1),  # move=1
                torch.logsumexp(action_logits[:, [3, 5]], dim=1),  # move=2
            ], dim=1)

            # Compute losses (with same weights as training)
            fire_loss = F.cross_entropy(fire_logits, fire_label, weight=fire_weights, label_smoothing=label_smoothing)
            move_loss = F.cross_entropy(move_logits, move_label, weight=move_weights, label_smoothing=label_smoothing)
            loss = fire_loss_weight * fire_loss + move_loss_weight * move_loss

            # Statistics
            action_pred = action_logits.argmax(dim=-1)
            fire_pred = fire_logits.argmax(dim=-1)
            move_pred = move_logits.argmax(dim=-1)

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
