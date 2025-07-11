"""
Linear Regression with Manual Gradient Descent Implementation
============================================================

This module implements linear regression from scratch using PyTorch's autograd system.
We'll solve the simple equation: y = w*x + b (in this case, y = 2*x)

Mathematical Background:
- Model: y = w*x + b
- Loss Function: MSE = (1/n) * Σ(y_pred - y_true)²
- Gradient: dL/dw = (2/n) * Σ(x * (y_pred - y_true))
- Update Rule: w = w - learning_rate * dL/dw

Learning Objectives:
1. Understand gradient computation and backpropagation
2. Implement manual gradient descent
3. Visualize loss convergence
4. Compare predicted vs actual values
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def create_dataset():
    """
    Create a simple linear dataset: y = 2*x
    
    Returns:
        tuple: (X, Y) tensors for training
    """
    X = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)
    Y = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=torch.float32)
    return X, Y

def predict(input_x, weight):
    """
    Linear prediction function: y = w*x
    
    Args:
        input_x (torch.Tensor): Input features
        weight (torch.Tensor): Model parameter
    
    Returns:
        torch.Tensor: Predicted values
    """
    return weight * input_x

def mse_loss(y_pred, y_true):
    """
    Mean Squared Error loss function
    
    Args:
        y_pred (torch.Tensor): Predicted values
        y_true (torch.Tensor): True values
    
    Returns:
        torch.Tensor: MSE loss
    """
    return ((y_pred - y_true) ** 2).mean()

def train_linear_regression(X, Y, learning_rate=0.001, epochs=100, verbose=True):
    """
    Train linear regression model using gradient descent
    
    Args:
        X (torch.Tensor): Input features
        Y (torch.Tensor): Target values
        learning_rate (float): Learning rate for gradient descent
        epochs (int): Number of training epochs
        verbose (bool): Whether to print training progress
    
    Returns:
        tuple: (trained_weight, loss_history)
    """
    # Initialize weight parameter with gradient tracking
    w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    
    # Track loss history for plotting
    loss_history = []
    
    print(f"Starting training with learning_rate={learning_rate}, epochs={epochs}")
    print("=" * 50)
    
    for epoch in range(1, epochs + 1):
        # Forward pass: compute predictions
        y_pred = predict(X, w)
        
        # Compute loss
        loss = mse_loss(y_pred, Y)
        
        # Backward pass: compute gradients
        # This traces back through the computation graph of how loss was calculated
        # Applies the chain rule at each step
        # Calculates the derivative of loss with respect to every tensor with requires_grad=True
        # Stores those derivatives (gradients) in the .grad attribute
        loss.backward()
        
        # Manual gradient descent update
        with torch.no_grad():
            w -= learning_rate * w.grad
        
        # Zero gradients for next iteration
        w.grad.zero_()
        
        # Store loss for plotting
        loss_history.append(loss.item())
        
        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f'Epoch {epoch:3d} ----> w: {w:.6f}, loss: {loss:.6f}')
    
    print("=" * 50)
    print(f"Training completed! Final weight: {w:.6f}")
    print(f"Target weight: 2.0 (since y = 2*x)")
    print(f"Final loss: {loss_history[-1]:.6f}")
    
    return w, loss_history

def evaluate_model(w, test_inputs):
    """
    Evaluate the trained model on test inputs
    
    Args:
        w (torch.Tensor): Trained weight
        test_inputs (list): List of test input values
    
    Returns:
        dict: Dictionary with test results
    """
    results = {}
    print("\nModel Evaluation:")
    print("-" * 30)
    
    for test_input in test_inputs:
        prediction = predict(torch.tensor(test_input, dtype=torch.float32), w)
        expected = 2 * test_input  # Since y = 2*x
        error = abs(prediction.item() - expected)
        
        results[test_input] = {
            'prediction': prediction.item(),
            'expected': expected,
            'error': error
        }
        
        print(f"f({test_input}) = {prediction.item():.3f}, "
              f"expected = {expected}, error = {error:.3f}")
    
    return results

def plot_results(X, Y, w, loss_history):
    """
    Plot training results including loss convergence and model fit
    
    Args:
        X (torch.Tensor): Training input features
        Y (torch.Tensor): Training target values
        w (torch.Tensor): Trained weight
        loss_history (list): List of loss values during training
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Loss convergence
    ax1.plot(loss_history, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training Loss Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Model fit
    X_np = X.numpy()
    Y_np = Y.numpy()
    
    # Plot original data
    ax2.scatter(X_np, Y_np, color='red', s=50, alpha=0.7, label='Training Data')
    
    # Plot model prediction
    X_extended = torch.linspace(0, 12, 100)
    Y_pred = predict(X_extended, w)
    ax2.plot(X_extended.numpy(), Y_pred.detach().numpy(), 
             'b-', linewidth=2, label=f'Learned: y = {w:.3f}x')
    
    # Plot true function
    ax2.plot(X_extended.numpy(), 2 * X_extended.numpy(), 
             'g--', linewidth=2, alpha=0.7, label='True: y = 2x')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Model Fit Comparison')
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
    
    # Train the model
    trained_weight, loss_history = train_linear_regression(
        X, Y, learning_rate=0.001, epochs=100, verbose=True
    )
    
    # Evaluate on test inputs
    test_inputs = [11, 13, 15, 20]
    results = evaluate_model(trained_weight, test_inputs)
    
    # Plot results
    plot_results(X, Y, trained_weight, loss_history)
    
    # Display final statistics
    print(f"\nFinal Model Statistics:")
    print(f"Learned weight: {trained_weight:.6f}")
    print(f"True weight: 2.0")
    print(f"Weight error: {abs(trained_weight.item() - 2.0):.6f}")
    print(f"Final loss: {loss_history[-1]:.6f}")

if __name__ == "__main__":
    main()