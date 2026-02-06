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
    def __init__(self, data_dir=None, subject=None, max_files=None, npz_files=None):
        # If npz_files is provided, use it directly
        if npz_files is not None:
            files_to_load = npz_files
            if max_files is not None:
                files_to_load = files_to_load[:max_files]
        else:
            # Find files from data_dir and subject
            if data_dir is None or subject is None:
                raise ValueError("Either npz_files or both data_dir and subject must be provided")

            self.data_dir = Path(data_dir)
            self.subject = subject

            # Load from specific subject directory
            subject_dir = self.data_dir / subject
            if not subject_dir.exists():
                raise ValueError(f"Subject directory not found: {subject_dir}")

            # Find all npz files
            files_to_load = sorted(glob.glob(str(subject_dir / '*.npz')))

            if max_files is not None:
                files_to_load = files_to_load[:max_files]

            print(f"Loading data from subject: {subject}")
            print(f"  Found {len(files_to_load)} files")

        # Load all data into memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        total_transitions = 0

        for npz_file in files_to_load:
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

        # Convert action to appropriate type
        action = torch.from_numpy(self.actions[idx])
        if action.dim() == 1 and len(action) > 1:  # One-hot encoded
            action = action.float()
        elif action.dim() == 0:  # Scalar index
            action = action.long()

        return {
            'state': state,  # (4, 84, 84) uint8
            'action': action,  # (6,) float or scalar long
            'reward': torch.tensor(self.rewards[idx], dtype=torch.float32),  # scalar
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


def create_train_val_dataloaders(data_dir, batch_size, subject, num_workers=4, val_split=0.1):
    """
    Create train and validation dataloaders by splitting files
    """
    # Find all npz files
    subject_dir = Path(data_dir) / subject
    if not subject_dir.exists():
        raise ValueError(f"Subject directory not found: {subject_dir}")

    npz_files = sorted(glob.glob(str(subject_dir / '*.npz')))
    n_files = len(npz_files)

    if n_files == 0:
        raise ValueError(f"No npz files found in {subject_dir}")

    # Calculate split
    n_val = max(1, int(n_files * val_split))
    n_train = n_files - n_val

    print(f"\nSplitting data:")
    print(f"  Total files: {n_files}")
    print(f"  Train files: {n_train}")
    print(f"  Val files: {n_val}")

    # Split files (last n_val files for validation)
    train_files = npz_files[:n_train]
    val_files = npz_files[n_train:]

    # Create datasets using the npz_files parameter
    print(f"\nLoading training data...")
    train_dataset = OfflineRLDataset(npz_files=train_files)

    print(f"\nLoading validation data...")
    val_dataset = OfflineRLDataset(npz_files=val_files)

    # Disable pin_memory for MPS (Apple Silicon GPU)
    use_pin_memory = torch.cuda.is_available()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader


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
