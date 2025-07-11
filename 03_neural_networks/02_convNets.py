"""
Convolutional Neural Networks - CIFAR-10 Image Classification
===========================================================

This module implements a CNN for image classification using the CIFAR-10 dataset.
Real-world application: Computer vision and image recognition tasks.

Architecture:
- Conv Layer 1: 3→32 channels, 3x3 kernel, ReLU + MaxPool
- Conv Layer 2: 32→64 channels, 3x3 kernel, ReLU + MaxPool  
- Conv Layer 3: 64→64 channels, 3x3 kernel, ReLU
- Fully Connected 1: 1024→64 neurons, ReLU + Dropout
- Fully Connected 2: 64→10 classes (output)

Mathematical Foundation:
- Convolution: (f * g)(x,y) = Σ Σ f(m,n) * g(x-m, y-n)
- Max Pooling: reduces spatial dimensions, retains important features
- Feature Maps: each filter detects specific patterns (edges, textures, etc.)
- Flattening: convert 3D feature maps to 1D for fully connected layers

Learning Objectives:
1. Understand convolutional operations and feature extraction
2. Learn CNN architecture design principles
3. Apply data augmentation and regularization
4. Evaluate model performance on real image data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=32, augment_data=True):
    """
    Create data loaders for CIFAR-10 dataset with optional data augmentation
    
    Args:
        batch_size (int): Batch size for training
        augment_data (bool): Whether to apply data augmentation
        
    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    # CIFAR-10 classes
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Data augmentation for training (helps prevent overfitting)
    if augment_data:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes: {classes}")
    print(f"Image size: 32x32x3 (RGB)")
    
    return train_loader, test_loader, classes

def visualize_samples(data_loader, classes, num_samples=25):
    """
    Visualize sample images from the dataset
    
    Args:
        data_loader: DataLoader for the dataset
        classes (list): List of class names
        num_samples (int): Number of samples to display
    """
    def denormalize(tensor):
        """Denormalize images for visualization"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    # Get a batch of images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    
    # Create grid of images
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    fig.suptitle('CIFAR-10 Sample Images', fontsize=16)
    
    for i in range(min(num_samples, len(images))):
        row, col = i // 5, i % 5
        
        # Denormalize and convert to numpy
        img = denormalize(images[i])
        img = torch.clamp(img, 0, 1)
        img_np = img.permute(1, 2, 0).numpy()
        
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f'{classes[labels[i]]}', fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

class CIFAR10CNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 Classification
    
    Architecture designed for 32x32 RGB images with 10 classes.
    Uses modern CNN design principles with batch normalization and dropout.
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        """
        Initialize the CNN model
        
        Args:
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout probability for regularization
        """
        super(CIFAR10CNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # Calculate flattened size: 128 channels * 4 * 4 = 2048
        self.flatten_size = 128 * 4 * 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU networks"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input images [batch_size, 3, 32, 32]
            
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        # Convolutional layers with batch norm and ReLU
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # [batch, 32, 16, 16]
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # [batch, 64, 8, 8]
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # [batch, 128, 4, 4]
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)  # [batch, 2048]
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)  # Final logits
        
        return x
    
    def get_feature_maps(self, x, layer_name='conv1'):
        """
        Extract feature maps from specified convolutional layer
        
        Args:
            x (torch.Tensor): Input images
            layer_name (str): Name of layer to extract features from
            
        Returns:
            torch.Tensor: Feature maps from specified layer
        """
        if layer_name == 'conv1':
            return F.relu(self.bn1(self.conv1(x)))
        elif layer_name == 'conv2':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            return F.relu(self.bn2(self.conv2(x)))
        elif layer_name == 'conv3':
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            return F.relu(self.bn3(self.conv3(x)))

def train_cnn_model(model, train_loader, test_loader, epochs=20, learning_rate=0.001, device='cpu'):
    """
    Train the CNN model with validation tracking
    
    Args:
        model (CIFAR10CNN): Model to train
        train_loader: Training data loader
        test_loader: Test data loader
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        device (str): Device to train on ('cpu' or 'cuda')
        
    Returns:
        tuple: (train_losses, train_accuracies, test_accuracies)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"Training CNN on {device}")
    print("=" * 60)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Epochs: {epochs}, Learning rate: {learning_rate}")
    print()
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Print progress every 500 batches
            if batch_idx % 500 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Validation phase
        test_accuracy = evaluate_model(model, test_loader, device, verbose=False)
        test_accuracies.append(test_accuracy)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch:2d}/{epochs} | '
              f'Train Loss: {epoch_loss:.4f} | '
              f'Train Acc: {epoch_accuracy:.4f} | '
              f'Test Acc: {test_accuracy:.4f} | '
              f'Time: {epoch_time:.1f}s')
    
    print("\nTraining completed!")
    return train_losses, train_accuracies, test_accuracies

def evaluate_model(model, test_loader, device='cpu', verbose=True):
    """
    Evaluate model performance on test set
    
    Args:
        model (CIFAR10CNN): Trained model
        test_loader: Test data loader
        device (str): Device to evaluate on
        verbose (bool): Whether to print detailed results
        
    Returns:
        float: Test accuracy
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    accuracy = correct / total
    
    if verbose:
        print(f"\nModel Evaluation Results")
        print("=" * 40)
        print(f"Overall Test Accuracy: {accuracy:.4f} ({correct}/{total})")
        print("\nPer-class Accuracy:")
        
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
        
        for i, class_name in enumerate(classes):
            if class_total[i] > 0:
                class_acc = class_correct[i] / class_total[i]
                print(f"  {class_name:8s}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
    
    return accuracy

def visualize_feature_maps(model, data_loader, device='cpu', layer_name='conv1'):
    """
    Visualize feature maps from a convolutional layer
    
    Args:
        model (CIFAR10CNN): Trained model
        data_loader: Data loader for sample images
        device (str): Device to run on
        layer_name (str): Layer to visualize
    """
    model.eval()
    
    # Get a sample image
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    sample_image = images[0:1].to(device)  # Take first image
    
    # Extract feature maps
    with torch.no_grad():
        feature_maps = model.get_feature_maps(sample_image, layer_name)
    
    # Move to CPU for visualization
    feature_maps = feature_maps.cpu().squeeze(0)  # Remove batch dimension
    num_filters = min(16, feature_maps.shape[0])  # Show up to 16 filters
    
    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f'Feature Maps from {layer_name.upper()}', fontsize=16)
    
    for i in range(num_filters):
        row, col = i // 4, i % 4
        feature_map = feature_maps[i]
        
        axes[row, col].imshow(feature_map, cmap='viridis')
        axes[row, col].set_title(f'Filter {i+1}', fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_results(train_losses, train_accuracies, test_accuracies):
    """
    Plot training metrics and results
    
    Args:
        train_losses (list): Training loss values
        train_accuracies (list): Training accuracy values
        test_accuracies (list): Test accuracy values
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, test_accuracies, 'r-', linewidth=2, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Accuracy Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def save_and_load_model(model, filepath='./models/cifar10_cnn.pth'):
    """
    Save and reload model state
    
    Args:
        model (CIFAR10CNN): Model to save
        filepath (str): Path to save model
        
    Returns:
        CIFAR10CNN: Loaded model
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
    
    # Load model
    loaded_model = CIFAR10CNN()
    loaded_model.load_state_dict(torch.load(filepath, map_location='cpu'))
    loaded_model.eval()
    print(f"Model loaded from {filepath}")
    
    return loaded_model

def main():
    """
    Main execution pipeline for CIFAR-10 CNN training and evaluation
    """
    print("CIFAR-10 Image Classification with Convolutional Neural Networks")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader, classes = get_data_loaders(batch_size=64, augment_data=True)
    
    # Visualize sample data
    print("\nVisualizing sample images...")
    visualize_samples(train_loader, classes)
    
    # Initialize model
    model = CIFAR10CNN(num_classes=10, dropout_rate=0.5)
    
    # Train model
    train_losses, train_accuracies, test_accuracies = train_cnn_model(
        model, train_loader, test_loader, epochs=15, learning_rate=0.001, device=device
    )
    
    # Evaluate final model
    final_accuracy = evaluate_model(model, test_loader, device)
    
    # Visualize feature maps
    print("\nVisualizing learned feature maps...")
    visualize_feature_maps(model, test_loader, device, layer_name='conv1')
    
    # Plot training results
    plot_training_results(train_losses, train_accuracies, test_accuracies)
    
    # Save model
    saved_model = save_and_load_model(model)
    
    # Final summary
    print(f"\nFinal Model Performance Summary")
    print("-" * 40)
    print(f"Test Accuracy: {final_accuracy:.4f}")
    print(f"Model Size: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Architecture: 3→32→64→128 Conv + 512→128→10 FC")
    print(f"Training completed successfully!")
    
    return model, (train_losses, train_accuracies, test_accuracies)

if __name__ == "__main__":
    trained_model, training_history = main()