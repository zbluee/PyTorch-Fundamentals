"""
Linear Regression using PyTorch's Built-in Modules
==================================================

This module implements linear regression using PyTorch's nn.Module and optimizers.
We'll solve: y = 3*x + 2

Key Differences from Manual Implementation:
- Uses nn.Linear for automatic weight/bias initialization
- Uses nn.MSELoss for loss computation
- Uses torch.optim.SGD for automatic gradient updates
- Demonstrates PyTorch's high-level API

Learning Objectives:
1. Understand PyTorch's nn.Module architecture
2. Learn to use built-in loss functions and optimizers
3. Compare with manual gradient descent approach
4. Visualize training progress and model performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def create_dataset():
    """
    Create dataset following: y = 3*x + 2
    
    Returns:
        tuple: (X, Y) tensors for training
    """
    X = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13]], 
                     dtype=torch.float32)
    Y = torch.tensor([[2], [5], [8], [11], [14], [17], [20], [23], [26], [29], [32], [35], [38], [41]], 
                     dtype=torch.float32)
    return X, Y

class LinearRegression(nn.Module):
    """
    Linear Regression Model using PyTorch's nn.Module
    
    This class encapsulates the linear transformation: y = wx + b
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initialize the linear regression model
        
        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
        """
        super(LinearRegression, self).__init__()
        # nn.Linear automatically initializes weights and bias
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Model predictions
        """
        return self.linear(x)
    
    def get_parameters(self):
        """
        Get model parameters (weight and bias)
        
        Returns:
            tuple: (weight, bias) values
        """
        weight = self.linear.weight.data[0][0].item()
        bias = self.linear.bias.data[0].item()
        return weight, bias

def train_model(model, X, Y, learning_rate=0.001, epochs=100, verbose=True):
    """
    Train the linear regression model
    
    Args:
        model (LinearRegression): Model to train
        X (torch.Tensor): Input features
        Y (torch.Tensor): Target values
        learning_rate (float): Learning rate for optimizer
        epochs (int): Number of training epochs
        verbose (bool): Whether to print training progress
        
    Returns:
        list: Loss history during training
    """
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    print(f"Starting training with lr={learning_rate}, epochs={epochs}")
    print("Target equation: y = 3*x + 2")
    print("=" * 60)
    
    for epoch in range(1, epochs + 1):
        # Forward pass
        y_pred = model(X)
        
        # Compute loss
        loss = criterion(y_pred, Y)
        
        # Backward pass and optimization
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
        optimizer.zero_grad()  # Clear gradients
        
        # Store loss for plotting
        loss_history.append(loss.item())
        
        # Print progress
        if verbose and (epoch % 50 == 0 or epoch == 1):
            w, b = model.get_parameters()
            print(f'Epoch {epoch:3d} ----> w: {w:.4f}, b: {b:.4f}, loss: {loss:.6f}')
    
    # Final results
    final_w, final_b = model.get_parameters()
    print("=" * 60)
    print(f"Training completed!")
    print(f"Learned equation: y = {final_w:.4f}*x + {final_b:.4f}")
    print(f"Target equation:  y = 3.0000*x + 2.0000")
    print(f"Weight error: {abs(final_w - 3.0):.6f}")
    print(f"Bias error: {abs(final_b - 2.0):.6f}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    
    return loss_history

def evaluate_model(model, test_inputs):
    """
    Evaluate model on test inputs
    
    Args:
        model (LinearRegression): Trained model
        test_inputs (list): List of test input values
        
    Returns:
        dict: Evaluation results
    """
    model.eval()  # Set model to evaluation mode
    results = {}
    
    print("\nModel Evaluation:")
    print("-" * 40)
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for test_input in test_inputs:
            x_test = torch.tensor([[test_input]], dtype=torch.float32)
            prediction = model(x_test).item()
            expected = 3 * test_input + 2  # True equation: y = 3*x + 2
            error = abs(prediction - expected)
            
            results[test_input] = {
                'prediction': prediction,
                'expected': expected,
                'error': error
            }
            
            print(f"f({test_input}) = {prediction:.3f}, "
                  f"expected = {expected:.3f}, error = {error:.3f}")
    
    return results

def plot_results(X, Y, model, loss_history):
    """
    Plot training results and model performance
    
    Args:
        X (torch.Tensor): Training input features
        Y (torch.Tensor): Training target values
        model (LinearRegression): Trained model
        loss_history (list): Loss values during training
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Loss convergence
    ax1.plot(loss_history, 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Loss Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add text with final loss
    ax1.text(0.7, 0.9, f'Final Loss: {loss_history[-1]:.6f}', 
             transform=ax1.transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot 2: Model fit
    X_np = X.numpy().flatten()
    Y_np = Y.numpy().flatten()
    
    # Plot training data
    ax2.scatter(X_np, Y_np, color='red', s=60, alpha=0.7, 
                label='Training Data', zorder=5)
    
    # Plot learned model
    X_extended = torch.linspace(0, 15, 100).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_extended)
    
    w, b = model.get_parameters()
    ax2.plot(X_extended.numpy(), Y_pred.numpy(), 
             'b-', linewidth=3, label=f'Learned: y = {w:.3f}x + {b:.3f}')
    
    # Plot true function
    ax2.plot(X_extended.numpy(), 3 * X_extended.numpy() + 2, 
             'g--', linewidth=2, alpha=0.8, label='True: y = 3x + 2')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Model Fit: PyTorch Linear Regression')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main training and evaluation pipeline
    """
    # Create dataset
    X, Y = create_dataset()
    n_samples, n_features = X.shape
    
    print(f"Dataset Info:")
    print(f"Samples: {n_samples}, Features: {n_features}")
    print(f"Input shape: {X.shape}, Output shape: {Y.shape}")
    
    # Initialize model
    model = LinearRegression(input_dim=n_features, output_dim=1)
    
    # Display initial parameters
    initial_w, initial_b = model.get_parameters()
    print(f"Initial parameters: w = {initial_w:.4f}, b = {initial_b:.4f}")
    
    # Train model
    loss_history = train_model(model, X, Y, learning_rate=0.001, epochs=500)
    
    # Evaluate on test inputs
    test_inputs = [15, 20, 25, 0, -5]
    results = evaluate_model(model, test_inputs)
    
    # Plot results
    plot_results(X, Y, model, loss_history)
    
    # Final model summary
    print(f"\nFinal Model Summary:")
    print(f"Model equation: y = {model.get_parameters()[0]:.4f}*x + {model.get_parameters()[1]:.4f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())} total")

if __name__ == "__main__":
    main()