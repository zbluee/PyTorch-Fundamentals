"""
Multi-Layer Perceptron - Email Category Classification
=====================================================

This module implements a multi-layer perceptron for multi-class classification using PyTorch.
Real-world application: Categorizing emails into multiple categories (Work, Personal, Spam, Promotions).

Architecture:
- Input Layer: 6 features 
- Hidden Layer 1: 64 neurons with ReLU activation and Dropout
- Hidden Layer 2: 32 neurons with ReLU activation and Dropout
- Output Layer: 4 neurons (one per class) with Softmax
- Regularization: Dropout layers to prevent overfitting

Mathematical Foundation:
- Hidden Layer 1: h1 = ReLU(W1 * x + b1)
- Dropout: h1_dropped = Dropout(h1, p=0.3)
- Hidden Layer 2: h2 = ReLU(W2 * h1_dropped + b2)
- Output: y = Softmax(W3 * h2_dropped + b3)
- Loss: Cross Entropy Loss

Learning Objectives:
1. Understand multi-layer neural networks
2. Learn multi-class classification
3. Apply dropout for regularization
4. Compare with single layer performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def create_email_dataset():
    """
    Create synthetic email dataset for 4-class classification.
    
    Features:
    - word_count: Number of words in email (10-300)
    - exclamation_marks: Number of '!' symbols (0-25)
    - capital_ratio: Ratio of capital letters (0-1)
    - money_mentioned: Binary flag for money keywords (0/1)
    - formal_words: Count of formal/business words (0-20)
    - time_sensitive: Binary flag for urgency keywords (0/1)
    
    Classes:
    0: Work emails
    1: Personal emails  
    2: Spam emails
    3: Promotional emails
    
    Returns:
        tuple: (X, Y, feature_names, class_names)
    """
    
    X = torch.tensor([
        # Work emails (class 0) - Professional, formal, moderate length
        [120, 0, 0.08, 0, 15, 0],  # Meeting invitation
        [200, 1, 0.12, 0, 18, 1],  # Urgent deadline
        [150, 0, 0.10, 0, 20, 0],  # Project update
        [180, 1, 0.09, 0, 16, 1],  # Important announcement
        [160, 0, 0.11, 0, 17, 0],  # Report submission
        [140, 1, 0.08, 0, 19, 1],  # Quarterly review
        [190, 0, 0.10, 0, 15, 0],  # Team collaboration
        [170, 1, 0.09, 0, 18, 0],  # Policy update
        
        # Personal emails (class 1) - Casual, friendly, varied length
        [80, 3, 0.25, 0, 2, 0],   # Friend catching up
        [60, 5, 0.35, 0, 1, 0],   # Excited personal news
        [100, 2, 0.20, 0, 3, 0],  # Family update
        [45, 4, 0.40, 0, 1, 0],   # Quick personal message
        [120, 3, 0.22, 0, 2, 0],  # Personal story
        [70, 6, 0.38, 0, 1, 0],   # Enthusiastic friend
        [90, 2, 0.28, 0, 2, 0],   # Personal invitation
        [55, 7, 0.42, 0, 1, 0],   # Excited personal update
        
        # Spam emails (class 2) - Short, aggressive, money-focused
        [30, 15, 0.85, 1, 0, 1],  # URGENT MONEY SCAM
        [25, 20, 0.90, 1, 0, 1],  # WIN CASH NOW
        [35, 12, 0.80, 1, 1, 1],  # FREE MONEY LIMITED
        [40, 18, 0.88, 1, 0, 1],  # CLICK FOR $$$
        [28, 22, 0.92, 1, 0, 1],  # MAKE MONEY FAST
        [32, 16, 0.86, 1, 1, 1],  # URGENT WINNER
        [38, 14, 0.82, 1, 0, 1],  # LOTTERY SCAM
        [26, 24, 0.94, 1, 0, 1],  # INHERITANCE SCAM
        
        # Promotional emails (class 3) - Marketing, moderate caps, some urgency
        [90, 4, 0.45, 0, 5, 1],   # Limited time sale
        [110, 6, 0.50, 0, 8, 1],  # Special discount offer
        [85, 3, 0.42, 0, 6, 0],   # Product announcement
        [75, 5, 0.48, 0, 7, 1],   # Flash sale alert
        [95, 4, 0.46, 0, 9, 1],   # Exclusive member deal
        [105, 7, 0.52, 0, 6, 1],  # Weekend promotion
        [80, 3, 0.44, 0, 8, 0],   # New product launch
        [100, 5, 0.49, 0, 7, 1],  # Seasonal sale
    ], dtype=torch.float32)
    
    # Labels for 4 classes
    Y = torch.tensor([
        # Work emails (0)
        0, 0, 0, 0, 0, 0, 0, 0,
        # Personal emails (1)  
        1, 1, 1, 1, 1, 1, 1, 1,
        # Spam emails (2)
        2, 2, 2, 2, 2, 2, 2, 2,
        # Promotional emails (3)
        3, 3, 3, 3, 3, 3, 3, 3
    ], dtype=torch.long)
    
    feature_names = ["word_count", "exclamation_marks", "capital_ratio", 
                    "money_mentioned", "formal_words", "time_sensitive"]
    class_names = ["Work", "Personal", "Spam", "Promotional"]
    
    return X, Y, feature_names, class_names

class EmailClassifierMLP(nn.Module):
    """
    Multi-Layer Perceptron for Email Classification
    
    Architecture:
    - Input: 6 features
    - Hidden Layer 1: 64 neurons + ReLU + Dropout(0.3)
    - Hidden Layer 2: 32 neurons + ReLU + Dropout(0.3)
    - Output: 4 classes + Softmax
    """
    
    def __init__(self, input_size, hidden1_size=64, hidden2_size=32, num_classes=4, dropout_rate=0.3):
        """
        Initialize the multi-layer perceptron
        
        Args:
            input_size (int): Number of input features
            hidden1_size (int): Size of first hidden layer
            hidden2_size (int): Size of second hidden layer
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout probability
        """
        super(EmailClassifierMLP, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Class logits (before softmax)
        """
        # First hidden layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer (no activation, CrossEntropyLoss includes softmax)
        x = self.fc3(x)
        
        return x
    
    def predict_proba(self, x):
        """
        Get prediction probabilities
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities

def train_mlp_classifier(model, X, Y, learning_rate=0.001, epochs=1000, verbose=True):
    """
    Train the multi-layer perceptron classifier
    
    Args:
        model (EmailClassifierMLP): Model to train
        X (torch.Tensor): Input features
        Y (torch.Tensor): Target labels
        learning_rate (float): Learning rate for optimization
        epochs (int): Number of training epochs
        verbose (bool): Whether to print training progress
        
    Returns:
        tuple: (loss_history, accuracy_history)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    loss_history = []
    accuracy_history = []
    
    if verbose:
        print("Training Multi-Layer Perceptron Email Classifier")
        print("=" * 60)
        print(f"Dataset: {X.shape[0]} emails, {X.shape[1]} features")
        print(f"Architecture: {X.shape[1]} → 64 → 32 → 4")
        print(f"Learning rate: {learning_rate}, Epochs: {epochs}")
        print()
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, Y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        model.eval()
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == Y).float().mean().item()
        
        # Store metrics
        loss_history.append(loss.item())
        accuracy_history.append(accuracy)
        
        # Print progress
        if verbose and (epoch % 200 == 0 or epoch == 1):
            print(f'Epoch {epoch:4d} | Loss: {loss.item():.6f} | Accuracy: {accuracy:.4f}')
    
    if verbose:
        print("\nTraining completed!")
        print(f"Final loss: {loss_history[-1]:.6f}")
        print(f"Final accuracy: {accuracy_history[-1]:.4f}")
    
    return loss_history, accuracy_history

def evaluate_mlp_model(model, test_cases, feature_names, class_names):
    """
    Evaluate the trained MLP model on test cases
    
    Args:
        model (EmailClassifierMLP): Trained model
        test_cases (list): List of (features, description) tuples
        feature_names (list): Names of input features
        class_names (list): Names of output classes
        
    Returns:
        list: Evaluation results
    """
    model.eval()
    results = []
    
    print("\nMulti-Layer Perceptron Evaluation")
    print("=" * 60)
    
    with torch.no_grad():
        for i, (features, description) in enumerate(test_cases):
            features_tensor = torch.tensor([features], dtype=torch.float32)
            
            # Get probabilities for all classes
            probabilities = model.predict_proba(features_tensor)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            
            result = {
                'description': description,
                'features': features,
                'predicted_class': predicted_class,
                'predicted_label': class_names[predicted_class],
                'confidence': confidence,
                'all_probabilities': probabilities.numpy()
            }
            results.append(result)
            
            print(f"Test Case {i+1}: {description}")
            print(f"  Predicted: {class_names[predicted_class]} (confidence: {confidence:.3f})")
            print(f"  Features: {dict(zip(feature_names, features))}")
            print(f"  All probabilities:")
            for j, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                print(f"    {class_name:12s}: {prob:.4f}")
            print()
    
    return results

def analyze_model_complexity(model, X):
    """
    Analyze model complexity and layer outputs
    
    Args:
        model (EmailClassifierMLP): Trained model
        X (torch.Tensor): Input data
    """
    print("Model Architecture Analysis")
    print("=" * 40)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Analyze layer sizes
    print("Layer Architecture:")
    print(f"Input Layer:    {model.fc1.in_features} features")
    print(f"Hidden Layer 1: {model.fc1.out_features} neurons (ReLU + Dropout)")
    print(f"Hidden Layer 2: {model.fc2.out_features} neurons (ReLU + Dropout)")
    print(f"Output Layer:   {model.fc3.out_features} classes (Softmax)")
    print()
    
    # Analyze activations with one sample
    model.eval()
    with torch.no_grad():
        x = X[0:1]  # Take first sample
        
        # Layer 1
        h1 = F.relu(model.fc1(x))
        print(f"Hidden Layer 1 activations (sample):")
        print(f"  Mean: {h1.mean().item():.4f}, Std: {h1.std().item():.4f}")
        print(f"  Active neurons: {(h1 > 0).sum().item()}/{h1.shape[1]}")
        
        # Layer 2  
        h2 = F.relu(model.fc2(h1))
        print(f"Hidden Layer 2 activations (sample):")
        print(f"  Mean: {h2.mean().item():.4f}, Std: {h2.std().item():.4f}")
        print(f"  Active neurons: {(h2 > 0).sum().item()}/{h2.shape[1]}")

def plot_training_results(loss_history, accuracy_history, X, Y, class_names):
    """
    Plot training metrics and data visualization
    
    Args:
        loss_history (list): Loss values during training
        accuracy_history (list): Accuracy values during training
        X (torch.Tensor): Training features
        Y (torch.Tensor): Training labels
        class_names (list): Names of output classes
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training Loss
    ax1.plot(loss_history, 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.set_title('Training Loss Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Training Accuracy
    ax2.plot(accuracy_history, 'g-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy Progress')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Plot 3: Feature Distribution by Class (Word Count vs Exclamation Marks)
    X_np = X.numpy()
    Y_np = Y.numpy()
    
    colors = ['blue', 'green', 'red', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for class_idx, (class_name, color, marker) in enumerate(zip(class_names, colors, markers)):
        mask = Y_np == class_idx
        ax3.scatter(X_np[mask, 0], X_np[mask, 1], 
                   c=color, s=60, alpha=0.7, label=class_name, marker=marker)
    
    ax3.set_xlabel('Word Count')
    ax3.set_ylabel('Exclamation Marks')
    ax3.set_title('Email Distribution: Word Count vs Exclamation Marks')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature Distribution (Capital Ratio vs Formal Words)
    for class_idx, (class_name, color, marker) in enumerate(zip(class_names, colors, markers)):
        mask = Y_np == class_idx
        ax4.scatter(X_np[mask, 2], X_np[mask, 4], 
                   c=color, s=60, alpha=0.7, label=class_name, marker=marker)
    
    ax4.set_xlabel('Capital Letter Ratio')
    ax4.set_ylabel('Formal Words Count')
    ax4.set_title('Email Distribution: Capital Ratio vs Formal Words')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main execution pipeline for multi-class email classification
    """
    print("Multi-Class Email Classification with Multi-Layer Perceptron")
    print("=" * 70)
    
    # Create dataset
    X, Y, feature_names, class_names = create_email_dataset()
    
    # Initialize model
    input_size = X.shape[1]
    model = EmailClassifierMLP(input_size, hidden1_size=64, hidden2_size=32, 
                              num_classes=len(class_names), dropout_rate=0.3)
    
    # Train model
    loss_history, accuracy_history = train_mlp_classifier(
        model, X, Y, learning_rate=0.001, epochs=1000
    )
    
    # Analyze model complexity
    analyze_model_complexity(model, X)
    
    # Define test cases
    test_cases = [
        ([150, 1, 0.10, 0, 18, 1], "Urgent work deadline"),
        ([70, 5, 0.35, 0, 2, 0], "Excited friend message"),
        ([30, 20, 0.90, 1, 0, 1], "Obvious spam scam"),
        ([90, 4, 0.45, 0, 6, 1], "Marketing promotion"),
        ([200, 0, 0.08, 0, 20, 0], "Detailed work report"),
        ([25, 25, 0.95, 1, 0, 1], "Extreme spam"),
        ([100, 3, 0.25, 0, 3, 0], "Personal story"),
        ([85, 6, 0.50, 0, 8, 1], "Flash sale alert")
    ]
    
    # Evaluate model
    results = evaluate_mlp_model(model, test_cases, feature_names, class_names)
    
    # Plot results
    plot_training_results(loss_history, accuracy_history, X, Y, class_names)
    
    # Final summary
    print("\nModel Performance Summary")
    print("-" * 30)
    print(f"Final training accuracy: {accuracy_history[-1]:.4f}")
    print(f"Architecture: {input_size} → 64 → 32 → {len(class_names)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dropout rate: 0.3")
    print(f"Classes: {', '.join(class_names)}")
    
    return model, results

if __name__ == "__main__":
    trained_model, evaluation_results = main()