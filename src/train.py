import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from src.model import MNISTModel
from src.config import TrainingConfig, ModelConfig
from datetime import datetime
import os
import logging
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(config: TrainingConfig = TrainingConfig()):
    # Set device
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=config.batch_size, 
                                             shuffle=True)
    
    # Create validation loader
    val_dataset = datasets.MNIST('./data', train=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                           batch_size=config.batch_size, 
                                           shuffle=False)
    
    # Model setup
    model_config = ModelConfig()
    model = MNISTModel(model_config).to(device)
    
    # Display model summary
    logger.info("Model Summary:")
    logger.info(model.get_summary(config.batch_size))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Metrics
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    # Training
    model.train()
    logger.info("Starting training...")
    
    # Create progress bar for epochs
    pbar = tqdm(range(config.epochs), desc="Epochs")
    for epoch in pbar:
        # Reset metrics
        losses.reset()
        accuracies.reset()
        
        # Create progress bar for batches
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config.epochs}", 
                         leave=False)
        
        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(target)
            
            # Update metrics
            losses.update(loss.item())
            accuracies.update(accuracy)
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.4f}'
            })
        
        # Validation phase
        model.eval()
        val_losses = AverageMeter()
        val_accuracies = AverageMeter()
        
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = correct / len(target)
                
                # Update metrics
                val_losses.update(loss.item())
                val_accuracies.update(accuracy)
                
                # Update progress bar
                val_pbar.set_postfix({
                    'val_loss': f'{val_losses.avg:.4f}',
                    'val_acc': f'{val_accuracies.avg:.4f}'
                })
        
        # Update epoch progress bar
        pbar.set_postfix({
            'train_loss': f'{losses.avg:.4f}',
            'train_acc': f'{accuracies.avg:.4f}',
            'val_loss': f'{val_losses.avg:.4f}',
            'val_acc': f'{val_accuracies.avg:.4f}'
        })
        
        model.train()
    
    # Save model with timestamp and metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = (f'mnist_model_{timestamp}_'
                 f'acc{val_accuracies.avg:.3f}.pth')
    save_path = os.path.join('models', model_name)
    os.makedirs('models', exist_ok=True)
    
    # Save model with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': losses.avg,
        'train_accuracy': accuracies.avg,
        'val_loss': val_losses.avg,
        'val_accuracy': val_accuracies.avg,
        'config': config,
    }, save_path)
    
    logger.info(f"Model saved to {save_path}")
    logger.info(f"Final Training Loss: {losses.avg:.4f}")
    logger.info(f"Final Training Accuracy: {accuracies.avg:.4f}")
    logger.info(f"Final Validation Loss: {val_losses.avg:.4f}")
    logger.info(f"Final Validation Accuracy: {val_accuracies.avg:.4f}")
    
    return save_path

if __name__ == "__main__":
    config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        epochs=1
    )
    train(config) 