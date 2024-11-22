# Efficient MNIST Model

A lightweight CNN implementation for MNIST digit classification with automated CI/CD pipeline.

## Model Specifications

- Parameters: < 25,000
- Training Accuracy: > 95% in 1 epoch
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)

## Model Architecture

- Conv2d(1 → 8, 3x3) + BatchNorm + ReLU + MaxPool
- Conv2d(8 → 16, 3x3) + BatchNorm + ReLU + MaxPool
- Dropout(0.25)
- Linear(16*7*7 → 10)

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest
- torchinfo
- tqdm
- numpy

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Trains the model
2. Verifies parameter count (< 25,000)
3. Validates training accuracy (> 95%)
4. Stores model artifacts

## License

MIT License

## Usage

1. Install dependencies: