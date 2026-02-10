import torch
import torch.nn as nn

class DQN_CNN(nn.Module):
    """
    DQN CNN backbone (Nature DQN architecture)
    Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
    Output: (batch, 3136) - feature vector
    """
    def __init__(self):
        super(DQN_CNN, self).__init__()

        self.cnn = nn.Sequential(
            # Conv1: (4, 84, 84) -> (32, 20, 20)
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            # Conv2: (32, 20, 20) -> (64, 9, 9)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Conv3: (64, 9, 9) -> (64, 7, 7)
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            # Flatten: (64, 7, 7) -> (3136,)
            nn.Flatten()
        )

    def forward(self, x):
        x = self.cnn(x)  # (batch, 3136)
        return x


class DQN(nn.Module):
    """
    Full DQN model (Nature DQN architecture)
    Architecture: CNN (3 conv layers) -> FC (512) -> FC (action_dim)
    """
    def __init__(self, action_dim=6):
        super(DQN, self).__init__()

        self.action_dim = action_dim

        # CNN layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # FC layers
        self.fc3 = nn.Linear(3136, 512)
        self.fc_out = nn.Linear(512, action_dim)

    def forward(self, x):
        # CNN forward
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten

        # FC forward
        x = torch.relu(self.fc3(x))
        q_values = self.fc_out(x)
        return q_values

    def get_action(self, state, deterministic=True):
        """Get action from state (compatible with eval.py)"""
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            q_values = self.forward(state)
            action = q_values.argmax(dim=-1)

        return action.item() if action.numel() == 1 else action


def load_pretrained_cnn(pretrained_path, freeze=True, freeze_conv12_only=False):
    cnn = DQN_CNN()

    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Extract policy_net if checkpoint has the full training state
        if 'policy_net' in checkpoint:
            state_dict = checkpoint['policy_net']
        else:
            state_dict = checkpoint

        # Map checkpoint keys (conv1, conv2, conv3) to DQN_CNN keys (cnn.0, cnn.2, cnn.4)
        # Filter out MLP layers (fc3, fc_out)
        key_mapping = {
            'conv1.weight': 'cnn.0.weight',
            'conv1.bias': 'cnn.0.bias',
            'conv2.weight': 'cnn.2.weight',
            'conv2.bias': 'cnn.2.bias',
            'conv3.weight': 'cnn.4.weight',
            'conv3.bias': 'cnn.4.bias',
        }

        # Create new state dict with mapped keys (CNN only, no MLP)
        mapped_state_dict = {}
        for old_key, new_key in key_mapping.items():
            if old_key in state_dict:
                mapped_state_dict[new_key] = state_dict[old_key]

        # Load only CNN parameters, ignore MLP (fc3, fc_out)
        cnn.load_state_dict(mapped_state_dict, strict=True)
        print(f"✓ Loaded pretrained CNN from {pretrained_path}")
        print(f"  Loaded {len(mapped_state_dict)} CNN parameter groups (MLP excluded)")

    except FileNotFoundError:
        print(f"✗ Pretrained CNN not found at {pretrained_path}")
        raise
    except Exception as e:
        print(f"✗ Error loading pretrained CNN: {e}")
        raise

    # Freeze encoder layers as specified
    if freeze:
        for param in cnn.parameters():
            param.requires_grad = False
        print(f"✓ CNN frozen (all conv layers frozen)")
    elif freeze_conv12_only:
        # Freeze Conv1 (cnn.0) and Conv2 (cnn.2), keep Conv3 (cnn.4) trainable
        for param in cnn.cnn[0].parameters():  # Conv1
            param.requires_grad = False
        for param in cnn.cnn[2].parameters():  # Conv2
            param.requires_grad = False
        # Explicitly ensure Conv3 is trainable (in case checkpoint had it frozen)
        for param in cnn.cnn[4].parameters():  # Conv3
            param.requires_grad = True
        print(f"✓ Conv1 and Conv2 frozen, Conv3 trainable")
    else:
        # FIX: Explicitly set requires_grad=True for all parameters
        # Checkpoint may have been saved with requires_grad=False, so we need to override
        for param in cnn.parameters():
            param.requires_grad = True
        print(f"✓ CNN unfrozen (all conv layers trainable)")

    return cnn
