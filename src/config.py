from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 512
    learning_rate: float = 0.01
    epochs: int = 1
    device: str = "cuda"

@dataclass
class ModelConfig:
    input_channels: int = 1
    num_classes: int = 10