import torch
import pytest
from src.model import MNISTModel
from src.config import ModelConfig, TrainingConfig
import os
from torchvision import datasets, transforms

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = MNISTModel(ModelConfig())
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, which exceeds the limit of 25000"

def test_model_accuracy():
    config = TrainingConfig(batch_size=512)
    model = MNISTModel(ModelConfig())
    
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not model_files:
        pytest.fail("No model file found")
    
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
    checkpoint = torch.load(os.path.join(models_dir, latest_model))
    
    # Check training accuracy from saved metrics
    train_accuracy = checkpoint['train_accuracy']
    assert train_accuracy > 0.95, f"Training accuracy {train_accuracy:.4f} is below 95%"