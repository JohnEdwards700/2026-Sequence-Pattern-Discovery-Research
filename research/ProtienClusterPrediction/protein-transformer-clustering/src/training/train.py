import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.data.dataset import ProteinDataset
from src.model.transformer import TransformerModel
from src.training.losses import ContrastiveLoss
from src.training.scheduler import get_scheduler
import yaml

def train_model(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_dataset = ProteinDataset(config['data']['train_file'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Initialize model
    model = TransformerModel(config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = get_scheduler(optimizer, config['training'])

    # Loss function
    criterion = ContrastiveLoss()

    model.train()
    for epoch in range(config['training']['epochs']):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch['input'].to(device), batch['target'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {epoch_loss/len(train_loader):.4f}")

    print("Training complete.")

if __name__ == "__main__":
    train_model('configs/default.yaml')