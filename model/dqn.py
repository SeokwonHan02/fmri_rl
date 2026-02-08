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


def load_pretrained_cnn(pretrained_path):
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

    # Freeze all CNN parameters (outputs 3136 features)
    for param in cnn.parameters():
        param.requires_grad = False

    print(f"✓ CNN frozen (outputs 3136 features, requires_grad=False)")

    return cnn
