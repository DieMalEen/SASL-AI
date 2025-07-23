import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import numpy as np
import os
import json

def extract_frames(video_path, num_frames=16):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle videos with fewer frames than requested
    if total_frames < num_frames:
        # If video has fewer frames, repeat the last frame to reach num_frames
        frame_idxs = list(range(total_frames))
        # Repeat the last frame index to fill remaining slots
        frame_idxs.extend([total_frames - 1] * (num_frames - total_frames))
    else:
        # Select evenly spaced frame indices
        frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the position to the selected frame
        ret, frame = cap.read()                # Read the frame
        if not ret:
            # If frame reading fails, use the last successfully read frame
            if frames:
                frame = cv2.cvtColor(np.array(frames[-1]), cv2.COLOR_RGB2BGR)
            else:
                # Create a black frame as fallback
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame = Image.fromarray(frame)                  # Convert to PIL Image
        frames.append(frame)                            # Add frame to list

    cap.release()  # Release the video capture object
    
    # Ensure we always return exactly num_frames frames
    while len(frames) < num_frames:
        frames.append(frames[-1])  # Duplicate last frame if needed
    
    return frames[:num_frames]  # Return exactly num_frames frames

def preprocess_video(video_path, transform, num_frames=16):
    # Extract frames from the video
    frames = extract_frames(video_path, num_frames)
    # Apply transformations to each frame and stack them into a tensor
    return torch.stack([transform(frame) for frame in frames])

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize frames to 224x224 pixels
    transforms.ToTensor(),          # Convert PIL Image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normalize using ImageNet means
                         [0.229, 0.224, 0.225]) # and standard deviations
])

# Define multiple augmentation strategies
augmentation_strategies = [
    # Strategy 1: Geometric transformations
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    # Strategy 2: Color/lighting changes
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    # Strategy 3: Combined transformations
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
]

def preprocess_video_with_strategy(video_path, strategy_idx, num_frames=16):
    # Extract frames from the video
    frames = extract_frames(video_path, num_frames)
    # Apply specific augmentation strategy
    transform = augmentation_strategies[strategy_idx]
    return torch.stack([transform(frame) for frame in frames])

class SASLDataset(Dataset):
    # Custom dataset for loading video frames and labels
    def __init__(self, videos, labels, transform):
        self.videos = videos      # List of video file paths
        self.labels = labels      # List of corresponding labels
        self.transform = transform # Transformations to apply to frames

    def __len__(self):
        # Return total number of videos
        return len(self.videos)

    def __getitem__(self, idx):
        # Load and preprocess frames from a video, return frames and label
        frames = preprocess_video(self.videos[idx], self.transform)
        label = self.labels[idx]
        return frames, label

class AdvancedSASLDataset(Dataset):
    def __init__(self, videos, labels, base_transform, augment=False, augment_factor=10):
        self.videos = videos
        self.labels = labels
        self.base_transform = base_transform
        self.augment = augment
        self.augment_factor = augment_factor
        
        if self.augment:
            self.expanded_videos = []
            self.expanded_labels = []
            self.augment_strategies = []
            
            for video, label in zip(videos, labels):
                # Add original
                self.expanded_videos.append(video)
                self.expanded_labels.append(label)
                self.augment_strategies.append(-1)  # -1 means no augmentation
                
                # Add augmented versions
                for i in range(augment_factor - 1):
                    self.expanded_videos.append(video)
                    self.expanded_labels.append(label)
                    self.augment_strategies.append(i % len(augmentation_strategies))
        else:
            self.expanded_videos = videos
            self.expanded_labels = labels
            self.augment_strategies = [-1] * len(videos)

    def __len__(self):
        return len(self.expanded_videos)

    def __getitem__(self, idx):
        video_path = self.expanded_videos[idx]
        label = self.expanded_labels[idx]
        strategy_idx = self.augment_strategies[idx]
        
        if strategy_idx == -1:
            # Use base transform (no augmentation)
            frames = preprocess_video(video_path, self.base_transform)
        else:
            # Use specific augmentation strategy
            frames = preprocess_video_with_strategy(video_path, strategy_idx)
            
        return frames, label
    
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
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device (GPU/CPU)

def load_dataset_from_folder(root_dir):
    # Load video paths and labels from dataset folders
    video_paths = []  # List to store video file paths
    labels = []       # List to store corresponding labels
    class_names = sorted(os.listdir(root_dir))  # Get sorted class names from dataset directory
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}  # Map class names to indices

    for class_name in class_names:
        class_folder = os.path.join(root_dir, class_name)  # Path to class folder
        for filename in os.listdir(class_folder):
            if filename.endswith(".mp4"):
                video_paths.append(os.path.join(class_folder, filename))  # Store video file path
                labels.append(class_to_idx[class_name])  # Store corresponding label

    return video_paths, labels, class_names  # Return lists for dataset construction

# Configuration flag to enable/disable validation
ENABLE_VALIDATION = False  # Set to True to enable validation, False to disable

# Load dataset and class names
root_dir = "dataset"  # Replace with your dataset path
video_paths, labels, class_names = load_dataset_from_folder(root_dir)  # Load dataset info

print(f"Classes: {class_names}")  # Print detected classes
print(f"Total videos: {len(video_paths)}")  # Print total number of videos

# Save class names to a JSON file immediately after loading the dataset
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("Class names saved to class_names.json")

# Split data only if validation is enabled
if ENABLE_VALIDATION:
    # Split data into training and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # Create dataset objects - TRAINING with augmentation, TESTING without
    # Use SASLDataset instead of AdvancedSASLDataset for original dataset
    train_dataset = AdvancedSASLDataset(train_paths, train_labels, transform, augment=True, augment_factor=4)
    test_dataset = AdvancedSASLDataset(test_paths, test_labels, transform, augment=False)
    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    print(f"Original training videos: {len(train_paths)}")
    print(f"Training dataset size with augmentation: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
else:
    # Use all data for training when validation is disabled - WITH AUGMENTATION
    # Use SASLDataset instead of AdvancedSASLDataset for original dataset
    train_dataset = AdvancedSASLDataset(video_paths, labels, transform, augment=True, augment_factor=4)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    print(f"Original videos: {len(video_paths)}")
    print(f"Training dataset size with augmentation: {len(train_dataset)}")

# Initialize model, loss function, and optimizer
model = CNN_LSTM(cnn=cnn_base, num_classes=len(class_names)).to(device)  # Model to device
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Optimizer

# Training loop
num_epochs = 10  # Number of training epochs
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    for frames, labels in train_loader:  # Iterate over batches
        frames, labels = frames.to(device), labels.to(device)  # Move data to device
        outputs = model(frames)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()  # Accumulate loss

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}")  # Print average loss per epoch

    # Validation phase (only if enabled)
    if ENABLE_VALIDATION:
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient calculation for validation
            for frames, labels in test_loader:
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                correct += (predicted == labels).sum().item()  # Count correct predictions
                total += labels.size(0)  # Total samples
        avg_val_loss = val_loss / len(test_loader)
        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")  # Print validation metrics

# Save the trained model to a file after training
torch.save(model.state_dict(), "sasl_model.pth")  # Save only the model weights
print("Model saved to sasl_model.pth")