import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained ResNet18 and prepare it as feature extractor
cnn_base = models.resnet18(pretrained=True)  # Load pre-trained ResNet18 model
cnn_base = nn.Sequential(*list(cnn_base.children())[:-1])  # Remove final fully connected layer
for param in cnn_base.parameters():
    param.requires_grad = False  # Freeze CNN weights

class CNN_LSTM(nn.Module):
    # Model combining CNN for spatial features and LSTM for temporal features
    def __init__(self, cnn, hidden_size=256, num_classes=20, num_layers=1):
        super(CNN_LSTM, self).__init__()
        self.cnn = cnn
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x: (batch, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)  # Flatten batch and sequence for CNN
        features = self.cnn(x)  # Extract features with CNN
        features = features.view(batch_size, seq_len, -1)  # Reshape for LSTM
        lstm_out, _ = self.lstm(features)  # Pass through LSTM
        out = self.fc(lstm_out[:, -1, :])  # Use output from last time step
        return out