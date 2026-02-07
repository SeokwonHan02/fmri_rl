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
    """
    Load pretrained CNN from checkpoint

    Args:
        pretrained_path: Path to the pretrained CNN checkpoint

    Returns:
        cnn: Loaded and frozen CNN model
    """
    cnn = DQN_CNN()

    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Load state dict directly (our format from download_and_extract_cnn.py)
        cnn.load_state_dict(checkpoint, strict=True)
        print(f"✓ Loaded pretrained CNN from {pretrained_path}")
        print(f"  Loaded {len(checkpoint)} parameter groups")

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
