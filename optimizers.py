"""
Optimization algorithms for gradient descent visualization.

Each optimizer takes:
- loss_function: A LossFunction object
- start_point: Initial (x, y) coordinates
- learning_rate: Step size
- num_iterations: Number of optimization steps

Returns:
- path: List of (x, y) points visited during optimization
"""

import numpy as np


def clip_gradient(grad, max_norm=10.0):
    """
    Clip gradient to prevent explosion.
    
    If ||grad|| > max_norm, scale it down to max_norm.
    This prevents NaN issues with aggressive learning rates or steep functions.
    """
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad


def vanilla_gradient_descent(loss_function, start_point, learning_rate, num_iterations):
    """
    Standard gradient descent (SGD).
    
    Update rule: θ_new = θ_old - learning_rate * ∇f(θ_old)
    
    Pure gradient descent - no momentum or adaptive learning rates.
    Can be slow on valleys and oscillate on steep surfaces.
    """
    path = [start_point]
    current_point = np.array(start_point, dtype=float)
    
    for _ in range(num_iterations):
        # Compute gradient at current position
        grad = loss_function.gradient(current_point[0], current_point[1])
        grad = clip_gradient(grad)
        
        # Take a step in the negative gradient direction
        current_point = current_point - learning_rate * grad
        path.append(tuple(current_point))
    
    return path


def momentum_gradient_descent(loss_function, start_point, learning_rate, num_iterations, momentum=0.9):
    """
    Gradient descent with momentum.
    
    Update rule:
        v_new = momentum * v_old + learning_rate * ∇f(θ_old)
        θ_new = θ_old - v_new
    
    Momentum accumulates gradients over time, helping to:
    - Speed up convergence in consistent directions
    - Dampen oscillations in ravines
    - Escape shallow local minima
    
    Common momentum values: 0.9, 0.95, 0.99
    """
    path = [start_point]
    current_point = np.array(start_point, dtype=float)
    velocity = np.zeros(2)  # Initialize velocity to zero
    
    for _ in range(num_iterations):
        grad = loss_function.gradient(current_point[0], current_point[1])
        grad = clip_gradient(grad)
        
        # Update velocity: blend old velocity with new gradient
        velocity = momentum * velocity + learning_rate * grad
        
        current_point = current_point - velocity
        path.append(tuple(current_point))
    
    return path


def adam_optimizer(loss_function, start_point, learning_rate, num_iterations, 
                   beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    Combines:
    - Momentum (first moment estimate)
    - RMSprop (second moment estimate for adaptive learning rates)
    
    Update rule:
        m_t = beta1 * m_{t-1} + (1-beta1) * ∇f      # First moment (mean)
        v_t = beta2 * v_{t-1} + (1-beta2) * ∇f²     # Second moment (variance)
        m_hat = m_t / (1 - beta1^t)                  # Bias correction
        v_hat = v_t / (1 - beta2^t)                  # Bias correction
        θ_new = θ_old - learning_rate * m_hat / (√v_hat + ε)
    
    Adam adapts learning rate per parameter based on gradient history.
    Very popular in deep learning - often works well with minimal tuning.
    
    Default hyperparameters (beta1=0.9, beta2=0.999) work well in most cases.
    """
    path = [start_point]
    current_point = np.array(start_point, dtype=float)

    m = np.zeros(2)  # First moment (momentum)
    v = np.zeros(2)  # Second moment (variance)
    
    for t in range(1, num_iterations + 1):  # t starts at 1 for bias correction
        # Compute gradient
        grad = loss_function.gradient(current_point[0], current_point[1])
        grad = clip_gradient(grad)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Compute bias-corrected moment estimates
        # This corrects for initialization bias (m and v start at 0)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update parameters
        # Divide by sqrt(v_hat) to get adaptive learning rate per dimension
        current_point = current_point - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        path.append(tuple(current_point))
    
    return path

OPTIMIZERS = {
    "Vanilla SGD": vanilla_gradient_descent,
    "Momentum": momentum_gradient_descent,
    "Adam": adam_optimizer,
}