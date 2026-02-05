import torch
import torch.nn as nn

class DQN_CNN(nn.Module):
    """
    DQN CNN backbone (Nature DQN architecture)
    Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
    Output: (batch, 3136) - flattened features
    """
    def __init__(self):
        super(DQN_CNN, self).__init__()

        self.conv = nn.Sequential(
            # Conv1: (4, 84, 84) -> (32, 20, 20)
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            # Conv2: (32, 20, 20) -> (64, 9, 9)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Conv3: (64, 9, 9) -> (64, 7, 7)
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Output size: 64 * 7 * 7 = 3136

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 3136)
        return x


def load_pretrained_cnn(pretrained_path):
    cnn = DQN_CNN()

    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Extract CNN weights from full DQN checkpoint
        if isinstance(checkpoint, dict):
            # Handle different checkpoint formats
            if 'policy_net' in checkpoint:
                state_dict = checkpoint['policy_net']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Map conv layer names and filter only conv layers (exclude MLP layers)
            cnn_state_dict = {}
            # Mapping: conv1 -> conv.0, conv2 -> conv.2, conv3 -> conv.4
            conv_mapping = {
                'conv1': 'conv.0',
                'conv2': 'conv.2',
                'conv3': 'conv.4'
            }

            for key, value in state_dict.items():
                # Check if it's a conv layer
                for old_name, new_name in conv_mapping.items():
                    if key.startswith(old_name):
                        new_key = key.replace(old_name, new_name)
                        # Remove 'module.' prefix if present
                        new_key = new_key.replace('module.', '')
                        cnn_state_dict[new_key] = value
                        break

            cnn.load_state_dict(cnn_state_dict, strict=True)
            print(f"Loaded pretrained CNN from {pretrained_path}")
            print(f"Loaded {len(cnn_state_dict)} parameters (conv layers only)")
        else:
            print(f"Unexpected checkpoint format")

    except FileNotFoundError:
        print(f"Pretrained DQN not found at {pretrained_path}")
    except Exception as e:
        print(f"Error loading pretrained CNN: {e}")

    # Always freeze CNN parameters
    for param in cnn.parameters():
        param.requires_grad = False

    return cnn
