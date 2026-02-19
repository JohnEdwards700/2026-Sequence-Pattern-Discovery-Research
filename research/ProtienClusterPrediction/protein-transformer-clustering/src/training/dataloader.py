from torch.utils.data import DataLoader, Dataset
import numpy as np

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label

class ProteinDataLoader:
    def __init__(self, sequences, labels, batch_size=32, shuffle=True):
        self.dataset = ProteinDataset(sequences, labels)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def get_loader(self):
        return self.dataloader