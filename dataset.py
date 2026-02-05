import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob

class OfflineRLDataset(Dataset):
    """
    Dataset for offline RL
    Loads all processed npz files and provides transitions (s, a, r, s', done)
    """
    def __init__(self, data_dir, subject='all', max_files=None):
        """
        Args:
            data_dir: Base directory containing processed data
            subject: Which subject to load ('all', 'sub_1', ..., 'sub_6')
            max_files: Maximum number of files to load (None = load all)
        """
        self.data_dir = Path(data_dir)
        self.subject = subject

        # Load from specific subject directory
        search_dirs = [self.data_dir / subject]
        if not search_dirs[0].exists():
            raise ValueError(f"Subject directory not found: {search_dirs[0]}")

        # Find all npz files
        npz_files = []
        for search_dir in search_dirs:
            npz_files.extend(sorted(glob.glob(str(search_dir / '*.npz'))))

        if max_files is not None:
            npz_files = npz_files[:max_files]

        print(f"Loading data from subject: {subject}")
        print(f"  Found {len(npz_files)} files")

        # Load all data into memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        total_transitions = 0

        for npz_file in npz_files:
            data = np.load(npz_file)

            self.states.append(data['state'])
            self.actions.append(data['action'])
            self.rewards.append(data['reward'])
            self.next_states.append(data['next_state'])
            self.dones.append(data['done'])

            total_transitions += len(data['state'])

        # Concatenate all data
        self.states = np.concatenate(self.states, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        self.next_states = np.concatenate(self.next_states, axis=0)
        self.dones = np.concatenate(self.dones, axis=0)

        print(f"Loaded {total_transitions} transitions")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # Transpose state from (H, W, C) to (C, H, W) for PyTorch
        state = torch.from_numpy(self.states[idx]).permute(2, 0, 1)  # (84, 84, 4) -> (4, 84, 84)
        next_state = torch.from_numpy(self.next_states[idx]).permute(2, 0, 1)  # (84, 84, 4) -> (4, 84, 84)

        return {
            'state': state,  # (4, 84, 84) uint8
            'action': torch.from_numpy(self.actions[idx]),  # (6,) float64
            'reward': torch.tensor(self.rewards[idx]),  # scalar
            'next_state': next_state,  # (4, 84, 84) uint8
            'done': torch.tensor(self.dones[idx], dtype=torch.float32)  # scalar
        }


def create_dataloader(data_dir, batch_size, subject, num_workers=4, shuffle=True, max_files=None):
    dataset = OfflineRLDataset(data_dir, subject=subject, max_files=max_files)

    # Disable pin_memory for MPS (Apple Silicon GPU)
    use_pin_memory = torch.cuda.is_available()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    return dataloader


if __name__ == '__main__':
    # Test dataset
    data_dir = '/Users/seokwon/research/fMRI_RL/processed_data/game_2_all'
    dataloader = create_dataloader(data_dir, batch_size=32, max_files=2)

    print("\nTesting DataLoader:")
    for batch in dataloader:
        print(f"Batch:")
        print(f"  State: {batch['state'].shape}, dtype={batch['state'].dtype}")
        print(f"  Action: {batch['action'].shape}, dtype={batch['action'].dtype}")
        print(f"  Reward: {batch['reward'].shape}, dtype={batch['reward'].dtype}")
        print(f"  Next state: {batch['next_state'].shape}, dtype={batch['next_state'].dtype}")
        print(f"  Done: {batch['done'].shape}, dtype={batch['done'].dtype}")
        break
