#Sri Rama Jayam
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class HeatEquation2DPINN(nn.Module):
    def __init__(self, num_layers=4, hidden_size=50):
        super(HeatEquation2DPINN, self).__init__()

        # Neural network layers
        layers = []
        # Input layer (x, y, t) -> 3 input features
        layers.append(nn.Linear(3, hidden_size))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers)

        # Heat equation parameter (thermal diffusivity)
        self.alpha = 0.01

    def forward(self, x, y, t):.000243,
        # Combine inputs into a single tensor
        inputs = torch.cat([x, y, t], dim=1)
        return self.net(inputs)

    def compute_derivatives(self, x, y, t):
        """Compute the derivatives needed for the 2D heat equation."""
        # Create variables that require gradients
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)

        # Forward pass
        u = self.forward(x, y, t)

        # First derivatives
        du_dx = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        du_dy = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        du_dt = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        # Second derivatives w.r.t. spatial coordinates
        d2u_dx2 = torch.autograd.grad(
            du_dx, x,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True
        )[0]

        d2u_dy2 = torch.autograd.grad(
            du_dy, y,
            grad_outputs=torch.ones_like(du_dy),
            create_graph=True
        )[0]

        return du_dt, d2u_dx2, d2u_dy2

    def pde_loss(self, x, y, t):
        """Compute the PDE residual loss for 2D heat equation."""
        # Get derivatives
        du_dt, d2u_dx2, d2u_dy2 = self.compute_derivatives(x, y, t)

        # 2D Heat equation: du/dt = α * (d²u/dx² + d²u/dy²)
        pde_residual = du_dt - self.alpha * (d2u_dx2 + d2u_dy2)

        return torch.mean(pde_residual ** 2)

    def boundary_loss(self, x_boundary, y_boundary, t_boundary, u_boundary):
        """Compute the boundary condition loss."""
        u_pred = self.forward(x_boundary, y_boundary, t_boundary)
        return torch.mean((u_pred - u_boundary) ** 2)

    def initial_loss(self, x_initial, y_initial, t_initial, u_initial):
        """Compute the initial condition loss."""
        u_pred = self.forward(x_initial, y_initial, t_initial)
        return torch.mean((u_pred - u_initial) ** 2)

# Training setup and helper functions
def generate_training_data(nx=20, ny=20, nt=10, x_range=(0, 1), y_range=(0, 1), t_range=(0, 1)):
    """Generate training points in the domain."""
    x = torch.linspace(x_range[0], x_range[1], nx)
    y = torch.linspace(y_range[0], y_range[1], ny)
    t = torch.linspace(t_range[0], t_range[1], nt)

    # Create meshgrid for all combinations
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')

    # Reshape to 2D tensors
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)

    return X_flat, Y_flat, T_flat

def generate_boundary_points(nx=10, ny=10, nt=5, x_range=(0, 1), y_range=(0, 1), t_range=(0, 1)):
    """Generate points on the boundaries."""
    # Time points
    t = torch.linspace(t_range[0], t_range[1], nt)

    # x-boundaries (y=0 and y=1)
    x_bound = torch.linspace(x_range[0], x_range[1], nx)
    t_mesh, x_mesh = torch.meshgrid(t, x_bound, indexing='ij')

    # Bottom boundary (y=0)
    x_bottom = x_mesh.reshape(-1, 1)
    t_bottom = t_mesh.reshape(-1, 1)
    y_bottom = torch.zeros_like(x_bottom)

    # Top boundary (y=1)
    x_top = x_mesh.reshape(-1, 1)
    t_top = t_mesh.reshape(-1, 1)
    y_top = torch.ones_like(x_top)

    # y-boundaries (x=0 and x=1)
    y_bound = torch.linspace(y_range[0], y_range[1], ny)
    t_mesh, y_mesh = torch.meshgrid(t, y_bound, indexing='ij')

    # Left boundary (x=0)
    y_left = y_mesh.reshape(-1, 1)
    t_left = t_mesh.reshape(-1, 1)
    x_left = torch.zeros_like(y_left)

    # Right boundary (x=1)
    y_right = y_mesh.reshape(-1, 1)
    t_right = t_mesh.reshape(-1, 1)
    x_right = torch.ones_like(y_right)

    # Combine all boundaries
    x_boundary = torch.cat([x_bottom, x_top, x_left, x_right], dim=0)
    y_boundary = torch.cat([y_bottom, y_top, y_left, y_right], dim=0)
    t_boundary = torch.cat([t_bottom, t_top, t_left, t_right], dim=0)

    return x_boundary, y_boundary, t_boundary

def gaussian_initial_condition(x, y, x0=0.5, y0=0.5, sigma=0.1):
    """Create a Gaussian initial temperature distribution."""
    return torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def train_pinn(model, num_epochs=2000, learning_rate=0.001):
    """Train the PINN model."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    # Generate interior training points
    x_train, y_train, t_train = generate_training_data()

    # Generate boundary points
    x_boundary, y_boundary, t_boundary = generate_boundary_points()
    u_boundary = torch.zeros_like(x_boundary)  # Dirichlet BC = 0 at all boundaries

    # Generate initial condition points
    x_initial = torch.linspace(0, 1, 50).reshape(-1, 1)
    y_initial = torch.linspace(0, 1, 50).reshape(-1, 1)
    X_init, Y_init = torch.meshgrid(x_initial.squeeze(), y_initial.squeeze(), indexing='ij')
    x_initial = X_init.reshape(-1, 1)
    y_initial = Y_init.reshape(-1, 1)
    t_initial = torch.zeros_like(x_initial)
    u_initial = gaussian_initial_condition(x_initial, y_initial)

    loss_history = []

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Compute PDE loss (interior points)
        pde_loss = model.pde_loss(x_train, y_train, t_train)

        # Compute boundary condition loss
        bc_loss = model.boundary_loss(x_boundary, y_boundary, t_boundary, u_boundary)

        # Compute initial condition loss
        ic_loss = model.initial_loss(x_initial, y_initial, t_initial, u_initial)

        # Total loss (weighted)
        total_loss = pde_loss + 10.0 * bc_loss + 10.0 * ic_loss

        # Backpropagation and optimization
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(total_loss.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item():.6f}, PDE: {pde_loss.item():.6f}, BC: {bc_loss.item():.6f}, IC: {ic_loss.item():.6f}')

    return loss_history

def plot_solution(model, t_value=0.5):
    """Plot the solution at a specific time."""
    nx = ny = 50
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    t_flat = torch.ones_like(x_flat) * t_value

    with torch.no_grad():
        u_pred = model(x_flat, y_flat, t_flat)
        U = u_pred.reshape(nx, ny)

    fig = plt.figure(figsize=(12, 5))

    # 2D contour plot
    ax1 = fig.add_subplot(121)
    contour = ax1.contourf(X.numpy(), Y.numpy(), U.numpy(), cmap=cm.viridis, levels=50)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Temperature at t = {t_value}')
    plt.colorbar(contour, ax=ax1, label='Temperature')

    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X.numpy(), Y.numpy(), U.numpy(), cmap=cm.viridis, linewidth=0, antialiased=False)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Temperature')
    ax2.set_title(f'Temperature at t = {t_value}')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create and train the model
    model = HeatEquation2DPINN()
    loss_history = train_pinn(model)

    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.grid(True)
    plt.show()

    # Plot solutions at different times
    for t in [0.0, 0.2, 0.5, 1.0]:
        plot_solution(model, t_value=t)
