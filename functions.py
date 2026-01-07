"""
Loss functions and their gradients for visualization.

Each function class implements:
- compute(x, y): Returns the function value
- gradient(x, y): Returns the gradient as [∂f/∂x, ∂f/∂y]
- bounds: Defines the region to visualize
- name: Human-readable name
"""

import numpy as np


class LossFunction:
    """Base class for loss functions."""
    
    def compute(self, x, y):
        """Compute function value at (x, y)."""
        raise NotImplementedError
    
    def gradient(self, x, y):
        """Compute gradient at (x, y). Returns [∂f/∂x, ∂f/∂y]."""
        raise NotImplementedError
    
    @property
    def bounds(self):
        """Return (x_min, x_max, y_min, y_max) for plotting."""
        raise NotImplementedError
    
    @property
    def name(self):
        """Return human-readable name."""
        raise NotImplementedError


class QuadraticBowl(LossFunction):
    """
    Simple convex quadratic function: f(x,y) = x² + y²
    
    Minimum at (0, 0) with value 0.
    This is the easiest optimization problem - gradient always points toward origin.
    """
    
    def compute(self, x, y):
        return x**2 + y**2
    
    def gradient(self, x, y):
        # ∂f/∂x = 2x, ∂f/∂y = 2y
        return np.array([2*x, 2*y])
    
    @property
    def bounds(self):
        return (-5, 5, -5, 5)
    
    @property
    def name(self):
        return "Quadratic Bowl (x² + y²)"


class RosenbrockFunction(LossFunction):
    """
    Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
    
    Minimum at (1, 1) with value 0.
    Famous for its narrow, curved valley - easy to find the valley, hard to converge.
    Tests optimizer's ability to navigate curved surfaces.
    """
    
    def compute(self, x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def gradient(self, x, y):
        # ∂f/∂x = -2(1-x) + 100·2(y-x²)·(-2x) = -2(1-x) - 400x(y-x²)
        # ∂f/∂y = 100·2(y-x²) = 200(y-x²)
        dx = -2*(1 - x) - 400*x*(y - x**2)
        dy = 200*(y - x**2)
        return np.array([dx, dy])
    
    @property
    def bounds(self):
        return (-2, 2, -1, 3)
    
    @property
    def name(self):
        return "Rosenbrock Function"


class SaddlePoint(LossFunction):
    """
    Saddle function: f(x,y) = x² - y²
    
    Has a saddle point at (0, 0) - not a minimum or maximum.
    Demonstrates non-convex optimization challenges.
    Gradient descent can get stuck depending on initialization.
    """
    
    def compute(self, x, y):
        return x**2 - y**2
    
    def gradient(self, x, y):
        # ∂f/∂x = 2x, ∂f/∂y = -2y
        return np.array([2*x, -2*y])
    
    @property
    def bounds(self):
        return (-5, 5, -5, 5)
    
    @property
    def name(self):
        return "Saddle Point (x² - y²)"


class BealeFunction(LossFunction):
    """
    Beale function: f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
    
    Minimum at (3, 0.5) with value 0.
    Has a narrow valley with sharp turns - very challenging for optimization.
    Tests optimizer robustness to non-convex, multimodal surfaces.
    """
    
    def compute(self, x, y):
        term1 = (1.5 - x + x*y)**2
        term2 = (2.25 - x + x*y**2)**2
        term3 = (2.625 - x + x*y**3)**2
        return term1 + term2 + term3
    
    def gradient(self, x, y):
        # Derivative is computed using chain rule
        term1 = 1.5 - x + x*y
        term2 = 2.25 - x + x*y**2
        term3 = 2.625 - x + x*y**3
        
        # ∂f/∂x
        dx = (2*term1*(-1 + y) + 
              2*term2*(-1 + y**2) + 
              2*term3*(-1 + y**3))
        
        # ∂f/∂y
        dy = (2*term1*x + 
              2*term2*x*2*y + 
              2*term3*x*3*y**2)
        
        return np.array([dx, dy])
    
    @property
    def bounds(self):
        return (-4.5, 4.5, -4.5, 4.5)
    
    @property
    def name(self):
        return "Beale Function"

FUNCTIONS = {
    "Quadratic Bowl": QuadraticBowl(),
    "Rosenbrock": RosenbrockFunction(),
    "Saddle Point": SaddlePoint(),
    "Beale": BealeFunction(),
}