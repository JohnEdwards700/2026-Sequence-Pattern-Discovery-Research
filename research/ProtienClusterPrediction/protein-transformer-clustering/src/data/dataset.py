import torch
from torch.utils.data import Dataset
import numpy as np

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Tokenize the sequence
        tokenized_sequence = self.tokenizer.encode(sequence, max_length=self.max_length, padding='max_length', truncation=True)

        return {
            'input_ids': torch.tensor(tokenized_sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }