# Gradient Descent Visualizer

An interactive web application for visualizing how different optimization algorithms navigate loss landscapes. Built with Python, NumPy, Matplotlib, and Streamlit.

![Gradient Descent Visualization](https://img.shields.io/badge/ML-Optimization-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green)

## 🎯 Features

- **4 Classic Loss Functions**: Quadratic Bowl, Rosenbrock, Saddle Point, Beale
- **3 Optimization Algorithms**: Vanilla SGD, Momentum, Adam
- **Interactive Controls**: Real-time parameter tuning with sliders
- **3D Visualization**: Beautiful surface plots with optimization paths
- **Metrics Dashboard**: Track convergence, loss reduction, and distance traveled

## 🚀 Quick Start

### Installation

1. **Clone or download this repository**

2. **Navigate to the project directory:**
   ```bash
   cd gradient_descent_viz
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install numpy==1.24.3 matplotlib==3.7.1 streamlit==1.28.0
   ```

### Running the Application

**Start the Streamlit app:**
```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

If it doesn't open automatically, navigate to the URL shown in your terminal.

## 📖 How to Use

### Basic Usage

1. **Select a loss function** from the dropdown (start with "Quadratic Bowl" for simplicity)
2. **Choose an optimizer** (try "Vanilla SGD" first)
3. **Adjust the learning rate** using the slider (start with 0.1)
4. **Set the number of iterations** (100 is a good default)
5. **Pick a starting position** using the X and Y sliders
6. **Watch the optimization path** appear on the 3D surface!

### Understanding the Visualization

- **Blue surface**: The loss landscape (height = loss value)
- **Red line**: Path taken by the optimizer
- **Green dot**: Starting position
- **Red star**: Final position
- **Metrics panel**: Shows convergence statistics

### Experimenting with Parameters

**Learning Rate:**
- **Too small (0.001)**: Slow convergence, many tiny steps
- **Just right (0.01-0.1)**: Smooth, efficient convergence
- **Too large (0.5+)**: Oscillation, overshooting, or divergence

**Different Optimizers:**
- **Vanilla SGD**: Simple, direct path
- **Momentum**: Smoother, faster on valleys (try on Rosenbrock!)
- **Adam**: Adaptive, often works well across all functions

**Different Functions:**
- **Quadratic Bowl**: Easy - should always converge
- **Rosenbrock**: Challenging valley - compare optimizers here!
- **Saddle Point**: Non-convex - watch different behaviors
- **Beale**: Complex - tests optimizer robustness

## 📂 Project Structure

```
gradient_descent_viz/
├── app.py              # Main Streamlit application
├── functions.py        # Loss function implementations
├── optimizers.py       # Optimization algorithm implementations
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🧮 Mathematical Background

### Gradient Descent

Gradient descent minimizes a function by iteratively moving in the direction of steepest descent:

```
θ_{t+1} = θ_t - α ∇f(θ_t)
```

Where:
- `θ` = parameters (x, y coordinates)
- `α` = learning rate (step size)
- `∇f` = gradient (direction of steepest ascent)

### Momentum

Adds velocity to smooth out updates:

```
v_{t+1} = β v_t + α ∇f(θ_t)
θ_{t+1} = θ_t - v_{t+1}
```

Where `β` (typically 0.9) controls momentum strength.

### Adam

Adaptive learning rates with first and second moment estimates:

```
m_t = β₁ m_{t-1} + (1-β₁) ∇f
v_t = β₂ v_{t-1} + (1-β₂) ∇f²
θ_{t+1} = θ_t - α m̂_t / (√v̂_t + ε)
```

Where `m̂` and `v̂` are bias-corrected estimates.

## 🎓 Learning Objectives

This project helps you understand:

1. **How gradient descent works** - visualize the algorithm in action
2. **Impact of learning rate** - see divergence, slow convergence, and sweet spots
3. **Optimizer differences** - compare vanilla SGD, Momentum, and Adam
4. **Loss landscape topology** - convex vs non-convex, valleys, saddle points
5. **Convergence behavior** - when and why algorithms succeed or struggle

## 🔧 Customization

### Adding New Loss Functions

Edit `functions.py`:

```python
class MyFunction(LossFunction):
    def compute(self, x, y):
        return # your function here
    
    def gradient(self, x, y):
        return np.array([dx, dy])  # partial derivatives
    
    @property
    def bounds(self):
        return (x_min, x_max, y_min, y_max)
    
    @property
    def name(self):
        return "My Custom Function"

# Add to dictionary
FUNCTIONS["My Function"] = MyFunction()
```

### Adding New Optimizers

Edit `optimizers.py`:

```python
def my_optimizer(loss_function, start_point, learning_rate, num_iterations):
    path = [start_point]
    current_point = np.array(start_point, dtype=float)
    
    for _ in range(num_iterations):
        grad = loss_function.gradient(current_point[0], current_point[1])
        # Your update rule here
        current_point = current_point - learning_rate * grad
        path.append(tuple(current_point))
    
    return path

# Add to dictionary
OPTIMIZERS["My Optimizer"] = my_optimizer
```

## 📝 Resume Description

**One-liner:**
> Built an interactive gradient descent visualizer comparing optimization algorithms (SGD, Adam, Momentum) with real-time parameter tuning and convergence analysis

**Detailed bullet points:**
- Implemented 4 classic optimization landscapes (Rosenbrock, Beale, etc.) with analytical gradients in NumPy
- Developed 3 optimization algorithms from scratch (Vanilla SGD, Momentum, Adam) with gradient clipping
- Created interactive 3D visualizations using Matplotlib and Streamlit, enabling real-time hyperparameter tuning
- Demonstrated deep understanding of optimization fundamentals including learning rate effects, momentum dynamics, and adaptive methods

## 🐛 Troubleshooting

**Issue**: App won't start
- **Solution**: Make sure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: Optimization diverges (path goes off screen)
- **Solution**: Reduce learning rate or use Adam optimizer (more stable)

**Issue**: Plots look weird
- **Solution**: Try different starting positions or reduce learning rate

**Issue**: "Module not found" error
- **Solution**: Make sure you're in the correct directory and dependencies are installed

## 📚 Further Reading

- [Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/)
- [An Overview of Gradient Descent Optimization Algorithms](https://arxiv.org/abs/1609.04747)
- [Why Momentum Really Works](https://distill.pub/2017/momentum/)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

---