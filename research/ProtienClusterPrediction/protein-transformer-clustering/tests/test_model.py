import pytest
import torch
from src.model.transformer import TransformerModel

def test_transformer_model_initialization():
    model = TransformerModel(input_dim=20, output_dim=10, num_heads=4, num_layers=2)
    assert model is not None
    assert isinstance(model, TransformerModel)

def test_transformer_forward_pass():
    model = TransformerModel(input_dim=20, output_dim=10, num_heads=4, num_layers=2)
    input_tensor = torch.rand(5, 10, 20)  # (batch_size, sequence_length, input_dim)
    output = model(input_tensor)
    assert output.shape == (5, 10, 10)  # (batch_size, sequence_length, output_dim)

def test_transformer_model_training():
    model = TransformerModel(input_dim=20, output_dim=10, num_heads=4, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    input_tensor = torch.rand(5, 10, 20)  # (batch_size, sequence_length, input_dim)
    target_tensor = torch.rand(5, 10, 10)  # (batch_size, sequence_length, output_dim)

    model.train()
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()

    assert loss.item() < 1.0  # Ensure loss decreases (arbitrary threshold for testing)