import torch
import torch.nn as nn
from torchinfo import torchinfo
from config import ModelConfig

class MNISTModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(config.input_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        #self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(16 * 7 * 7, config.num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.batch_norm2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        #x = self.dropout(x)
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc1(x)
        return x

    def get_summary(self, batch_size=1):
        return torchinfo.summary(self, input_size=(batch_size, 1, 28, 28), 
                               col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                               verbose=2)