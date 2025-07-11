"""
Single Layer Perceptron - Email Spam Detection
==============================================

This module implements a single layer perceptron for binary classification using PyTorch.
Real-world application: Email spam detection based on text features.

Architecture:
- Input Layer: 4 features (word_count, exclamation_marks, capital_ratio, money_mentioned)
- Single Linear Layer: 4 → 1 with weights and bias
- Activation: Sigmoid function for probability output
- Loss: Binary Cross Entropy (BCE)

Mathematical Foundation:
- Linear Transformation: z = w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + b
- Sigmoid Activation: σ(z) = 1 / (1 + e^(-z))
- Output: P(spam) = σ(z), where 0 ≤ P(spam) ≤ 1

Learning Objectives:
1. Understand single neuron classification
2. Learn feature engineering for text classification
3. Interpret model weights and predictions
4. Apply binary classification to real problems
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def create_spam_dataset():
    """
    Create synthetic email spam dataset with realistic features.
    
    Features:
    - word_count: Number of words in email (10-200)
    - exclamation_marks: Number of '!' symbols (0-25)
    - capital_ratio: Ratio of capital letters (0-1)
    - money_mentioned: Binary flag for money-related keywords (0/1)
    
    Returns:
        tuple: (X, Y, feature_names) where X is features, Y is labels
    """
    # Feature engineering based on real spam characteristics
    X = torch.tensor([
        # Regular emails (label 0) - Normal communication patterns
        [50, 0, 0.10, 0],   # Professional email
        [120, 1, 0.15, 0],  # Casual email with mild emphasis
        [80, 0, 0.12, 0],   # Work correspondence
        [200, 2, 0.18, 0],  # Enthusiastic friend email
        [45, 0, 0.08, 0],   # Brief message
        [90, 1, 0.11, 0],   # Regular communication
        [150, 0, 0.13, 0],  # Detailed email
        [65, 1, 0.14, 0],   # Normal email with emphasis
        [110, 0, 0.09, 0],  # Standard business email
        [75, 2, 0.16, 0],   # Excited but legitimate email
        
        # Spam emails (label 1) - Aggressive marketing patterns
        [30, 8, 0.70, 1],   # "URGENT!!! MONEY OFFER!!!"
        [25, 12, 0.80, 1],  # "WIN CASH NOW!!!"
        [40, 6, 0.65, 1],   # "FREE MONEY LIMITED TIME!"
        [35, 10, 0.75, 1],  # "CLICK HERE FOR $$$!!!"
        [45, 7, 0.60, 1],   # "AMAZING OFFER! MONEY!"
        [20, 15, 0.90, 1],  # "MAKE MONEY FAST!!!!!!"
        [55, 5, 0.55, 1],   # "You've won! Claim money!"
        [28, 18, 0.85, 1],  # "URGENT!!! ACT NOW!!!"
        [42, 9, 0.68, 1],   # "LIMITED TIME!!! CASH!!!"
        [33, 11, 0.78, 1],  # "WINNER!!! COLLECT $$$!"
    ], dtype=torch.float32)
    
    Y = torch.tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0],  # Regular emails
                      [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype=torch.float32)  # Spam emails
    
    feature_names = ["word_count", "exclamation_marks", "capital_ratio", "money_mentioned"]
    
    return X, Y, feature_names

class SpamDetector(nn.Module):
    """
    Single Layer Perceptron for Email Spam Detection
    
    This neural network consists of:
    - One linear layer (fully connected)
    - Sigmoid activation function
    - Binary classification output
    """
    
    def __init__(self, input_size):
        """
        Initialize the spam detector model
        
        Args:
            input_size (int): Number of input features
        """
        super(SpamDetector, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights with small random values for better convergence
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Probability of being spam (0-1)
        """
        linear_output = self.linear(x)
        probability = self.sigmoid(linear_output)
        return probability
    
    def get_feature_importance(self, feature_names):
        """
        Extract and interpret learned feature weights
        
        Args:
            feature_names (list): Names of the input features
            
        Returns:
            dict: Feature importance scores
        """
        weights = self.linear.weight.data[0].numpy()
        bias = self.linear.bias.data[0].item()
        
        importance = {}
        for i, name in enumerate(feature_names):
            importance[name] = weights[i]
        importance['bias'] = bias
        
        return importance

def train_spam_detector(model, X, Y, learning_rate=0.01, epochs=2000, verbose=True):
    """
    Train the spam detection model
    
    Args:
        model (SpamDetector): Model to train
        X (torch.Tensor): Input features
        Y (torch.Tensor): Target labels
        learning_rate (float): Learning rate for optimization
        epochs (int): Number of training epochs
        verbose (bool): Whether to print training progress
        
    Returns:
        list: Loss history during training
    """
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    if verbose:
        print("Training Email Spam Detector")
        print("=" * 50)
        print(f"Dataset: {X.shape[0]} emails, {X.shape[1]} features")
        print(f"Learning rate: {learning_rate}, Epochs: {epochs}")
        print()
    
    for epoch in range(1, epochs + 1):
        # Forward pass
        y_pred = model(X)
        
        # Compute loss
        loss = criterion(y_pred, Y)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Store loss for plotting
        loss_history.append(loss.item())
        
        # Print progress
        if verbose and (epoch % 500 == 0 or epoch == 1):
            print(f'Epoch {epoch:4d} | Loss: {loss.item():.6f}')
    
    if verbose:
        print("\nTraining completed!")
        print(f"Final loss: {loss_history[-1]:.6f}")
    
    return loss_history

def evaluate_model(model, test_cases, feature_names):
    """
    Evaluate the trained model on test cases
    
    Args:
        model (SpamDetector): Trained model
        test_cases (list): List of (features, description) tuples
        feature_names (list): Names of the input features
        
    Returns:
        list: Evaluation results
    """
    model.eval()
    results = []
    
    print("\nModel Evaluation on Test Cases")
    print("=" * 60)
    
    with torch.no_grad():
        for i, (features, description) in enumerate(test_cases):
            features_tensor = torch.tensor([features], dtype=torch.float32)
            prediction = model(features_tensor)
            
            prob_spam = prediction.item()
            classification = "SPAM" if prob_spam > 0.5 else "NORMAL"
            confidence = prob_spam if prob_spam > 0.5 else 1 - prob_spam
            
            result = {
                'description': description,
                'features': features,
                'probability': prob_spam,
                'classification': classification,
                'confidence': confidence
            }
            results.append(result)
            
            print(f"Test Case {i+1}: {description}")
            print(f"  Classification: {classification} (confidence: {confidence:.3f})")
            print(f"  Features: {dict(zip(feature_names, features))}")
            print(f"  Spam probability: {prob_spam:.4f}")
            print()
    
    return results

def analyze_feature_importance(model, feature_names):
    """
    Analyze and display feature importance
    
    Args:
        model (SpamDetector): Trained model
        feature_names (list): Names of the input features
    """
    importance = model.get_feature_importance(feature_names)
    
    print("Feature Importance Analysis")
    print("=" * 40)
    print("Positive weights → Higher values increase spam probability")
    print("Negative weights → Higher values decrease spam probability")
    print()
    
    for feature, weight in importance.items():
        if feature != 'bias':
            direction = "↑ Spam indicator" if weight > 0 else "↓ Normal indicator"
            print(f"{feature:18s}: {weight:8.4f} {direction}")
        else:
            print(f"{'bias':18s}: {weight:8.4f}")

def plot_training_results(loss_history, X, Y, model):
    """
    Plot training loss and decision boundary visualization
    
    Args:
        loss_history (list): Loss values during training
        X (torch.Tensor): Training features
        Y (torch.Tensor): Training labels
        model (SpamDetector): Trained model
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training loss
    ax1.plot(loss_history, 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Cross Entropy Loss')
    ax1.set_title('Training Loss Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add final loss annotation
    final_loss = loss_history[-1]
    ax1.annotate(f'Final Loss: {final_loss:.4f}', 
                xy=(len(loss_history)-1, final_loss),
                xytext=(len(loss_history)*0.7, final_loss*2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    # Plot 2: Feature scatter (exclamation marks vs capital ratio)
    X_np = X.numpy()
    Y_np = Y.numpy().flatten()
    
    # Separate spam and normal emails
    normal_idx = Y_np == 0
    spam_idx = Y_np == 1
    
    ax2.scatter(X_np[normal_idx, 1], X_np[normal_idx, 2], 
               c='blue', s=60, alpha=0.7, label='Normal', marker='o')
    ax2.scatter(X_np[spam_idx, 1], X_np[spam_idx, 2], 
               c='red', s=60, alpha=0.7, label='Spam', marker='^')
    
    ax2.set_xlabel('Exclamation Marks')
    ax2.set_ylabel('Capital Letter Ratio')
    ax2.set_title('Email Classification: Exclamation Marks vs Capital Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main execution pipeline for spam detection training and evaluation
    """
    print("Email Spam Detection with Single Layer Perceptron")
    print("=" * 60)
    
    # Create dataset
    X, Y, feature_names = create_spam_dataset()
    
    # Initialize model
    input_size = X.shape[1]
    model = SpamDetector(input_size)
    
    # Train model
    loss_history = train_spam_detector(model, X, Y, learning_rate=0.01, epochs=2000)
    
    # Analyze feature importance
    analyze_feature_importance(model, feature_names)
    
    # Define test cases
    test_cases = [
        ([100, 1, 0.14, 0], "Business email: moderate length, minimal emphasis"),
        ([25, 20, 0.85, 1], "Suspicious: short, excessive punctuation, money"),
        ([75, 0, 0.10, 0], "Professional: moderate length, no emphasis"),
        ([15, 25, 0.95, 1], "Obvious spam: short, excessive caps and punctuation"),
        ([150, 3, 0.20, 0], "Enthusiastic but legitimate: long, some emphasis"),
        ([35, 15, 0.88, 1], "Classic spam pattern: short, aggressive, money-focused")
    ]
    
    # Evaluate model
    results = evaluate_model(model, test_cases, feature_names)
    
    # Plot results
    plot_training_results(loss_history, X, Y, model)
    
    # Model summary
    print("\nModel Architecture Summary")
    print("-" * 30)
    print(f"Input features: {input_size}")
    print(f"Output: 1 (spam probability)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Activation: Sigmoid")
    print(f"Loss function: Binary Cross Entropy")
    
    return model, results

if __name__ == "__main__":
    trained_model, evaluation_results = main()