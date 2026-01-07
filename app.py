"""
Gradient Descent Visualizer

Interactive tool to visualize how different optimization algorithms
navigate loss landscapes.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from functions import FUNCTIONS
from optimizers import OPTIMIZERS


def create_surface_mesh(loss_function, num_points=50):
    """
    Create a mesh grid for 3D surface plotting.
    
    Args:
        loss_function: LossFunction object
        num_points: Resolution of the mesh (higher = smoother but slower)
    
    Returns:
        X, Y, Z: Mesh grid coordinates and function values
    """
    x_min, x_max, y_min, y_max = loss_function.bounds
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x, y)
    
    # Compute function value at each grid point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = loss_function.compute(X[i, j], Y[i, j])
    
    return X, Y, Z


def plot_optimization(loss_function, path, start_point):
    """
    Create 3D visualization of loss surface and optimization path.
    
    Args:
        loss_function: LossFunction object
        path: List of (x, y) points from optimizer
        start_point: Initial (x, y) coordinates
    
    Returns:
        matplotlib figure object
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = create_surface_mesh(loss_function, num_points=50)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none', antialiased=True)
    
    # Extract path coordinates and compute function values along path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    path_z = [loss_function.compute(p[0], p[1]) for p in path]
    
    ax.plot(path_x, path_y, path_z, 'r-', linewidth=2, label='Optimization Path', alpha=0.9)
    
    # Mark start and end points
    start_z = loss_function.compute(start_point[0], start_point[1])
    ax.scatter([start_point[0]], [start_point[1]], [start_z], 
               color='green', s=100, marker='o', label='Start', zorder=5)
    
    end_point = path[-1]
    end_z = path_z[-1]
    ax.scatter([end_point[0]], [end_point[1]], [end_z], 
               color='red', s=200, marker='*', label='End', zorder=5)
    
    # Labeling
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Loss', fontsize=10)
    ax.set_title(f'{loss_function.name}\nOptimization Path Visualization', 
                 fontsize=12, pad=20)
    

    ax.view_init(elev=25, azim=45)
    ax.legend(loc='upper right')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    return fig


def compute_metrics(loss_function, path):
    """
    Compute optimization metrics for analysis.
    
    Args:
        loss_function: LossFunction object
        path: List of (x, y) points
    
    Returns:
        dict: Metrics including final loss, distance traveled, etc.
    """
    # Initial and final loss
    initial_loss = loss_function.compute(path[0][0], path[0][1])
    final_loss = loss_function.compute(path[-1][0], path[-1][1])
    loss_reduction = initial_loss - final_loss
    
    # Total distance traveled
    total_distance = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        total_distance += np.sqrt(dx**2 + dy**2)
    
    # Final position
    final_x, final_y = path[-1]
    
    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction': loss_reduction,
        'reduction_percent': (loss_reduction / initial_loss * 100) if initial_loss != 0 else 0,
        'total_distance': total_distance,
        'final_position': (final_x, final_y),
        'num_steps': len(path) - 1
    }


def main():
    """Main Streamlit application."""

    st.set_page_config(
        page_title="Gradient Descent Visualizer",
        layout="wide"
    )
    
    st.title("Gradient Descent Visualizer")
    st.markdown("""
    Visualize how different optimization algorithms navigate loss landscapes.
    Adjust parameters and see how gradient descent finds the minimum of various functions.
    """)
    
    st.sidebar.header("Configuration")
    
    # Function selection
    st.sidebar.subheader("Loss Function")
    function_name = st.sidebar.selectbox(
        "Select function:",
        list(FUNCTIONS.keys()),
        help="Different functions have different optimization challenges"
    )
    loss_function = FUNCTIONS[function_name]
    
    # Optimizer selection
    st.sidebar.subheader("Optimizer")
    optimizer_name = st.sidebar.selectbox(
        "Select optimizer:",
        list(OPTIMIZERS.keys()),
        help="Compare different optimization algorithms"
    )
    optimizer = OPTIMIZERS[optimizer_name]
    
    st.sidebar.subheader("Hyperparameters")
    
    learning_rate = st.sidebar.slider(
        "Learning Rate:",
        min_value=0.001,
        max_value=1.0,
        value=0.1,
        step=0.001,
        format="%.3f",
        help="Step size for each iteration. Too large = divergence, too small = slow convergence"
    )
    
    num_iterations = st.sidebar.slider(
        "Number of Iterations:",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="How many optimization steps to take"
    )
    
    st.sidebar.subheader("Starting Position")
    x_min, x_max, y_min, y_max = loss_function.bounds
    
    start_x = st.sidebar.slider(
        "Initial X:",
        min_value=float(x_min),
        max_value=float(x_max),
        value=float(x_max * 0.7),  # Start away from center for interesting paths
        step=0.1
    )
    
    start_y = st.sidebar.slider(
        "Initial Y:",
        min_value=float(y_min),
        max_value=float(y_max),
        value=float(y_max * 0.7),
        step=0.1
    )
    
    start_point = (start_x, start_y)
    
    st.sidebar.markdown("---")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Optimization Visualization")
        
        # Run optimization
        with st.spinner("Running optimization..."):
            path = optimizer(loss_function, start_point, learning_rate, num_iterations)
            fig = plot_optimization(loss_function, path, start_point)
            st.pyplot(fig)
            plt.close()
    
    with col2:
        st.subheader("Metrics")
        
        # Compute and display metrics
        metrics = compute_metrics(loss_function, path)
        
        st.metric("Initial Loss", f"{metrics['initial_loss']:.4f}")
        st.metric("Final Loss", f"{metrics['final_loss']:.4f}")
        st.metric("Loss Reduction", 
                 f"{metrics['loss_reduction']:.4f}",
                 delta=f"-{metrics['reduction_percent']:.1f}%")
        
        st.markdown("---")
        
        st.metric("Total Distance Traveled", f"{metrics['total_distance']:.2f}")
        st.metric("Number of Steps", metrics['num_steps'])
        
        st.markdown("---")
        
        st.write("**Final Position:**")
        st.write(f"X: {metrics['final_position'][0]:.4f}")
        st.write(f"Y: {metrics['final_position'][1]:.4f}")
    
    # Additional information section
    st.markdown("---")
    
    with st.expander("ℹAbout the Functions"):
        st.markdown("""
        **Quadratic Bowl**: Simplest convex function. Gradient always points to minimum.
        
        **Rosenbrock**: Famous for its narrow valley. Easy to find but hard to converge.
        
        **Saddle Point**: Non-convex with a saddle at origin. Shows optimization challenges.
        
        **Beale**: Complex landscape with sharp valleys. Tests optimizer robustness.
        """)
    
    with st.expander("About the Optimizers"):
        st.markdown("""
        **Vanilla SGD**: Basic gradient descent. Simple but can be slow.
        
        **Momentum**: Accumulates gradients over time. Helps with valleys and oscillations.
        
        **Adam**: Adaptive learning rates per parameter. Usually works well with minimal tuning.
        """)


if __name__ == "__main__":
    main()