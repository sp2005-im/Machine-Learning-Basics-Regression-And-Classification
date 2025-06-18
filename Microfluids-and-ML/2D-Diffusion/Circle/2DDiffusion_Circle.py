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

    def forward(self, x, y, t):
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
        pde_residual = du_dt + self.alpha * (d2u_dx2 + d2u_dy2)

        return torch.mean(pde_residual ** 2)

    def boundary_loss(self, x_boundary, y_boundary, t_boundary, u_boundary):
        """Compute the boundary condition loss."""
        u_pred = self.forward(x_boundary, y_boundary, t_boundary)
        return torch.mean((u_pred - u_boundary) ** 2)

    def initial_loss(self, x_initial, y_initial, t_initial, u_initial):
        """Compute the initial condition loss."""
        u_pred = self.forward(x_initial, y_initial, t_initial)
        return torch.mean((u_pred - u_initial) ** 2)

def generate_circle_domain(n_r=15, n_theta=30, n_t=10, radius=1.0, t_range=(0, 1)):
    """Generate training points in a circular domain."""
    # Radial coordinates
    r = torch.linspace(0, radius, n_r)
    # Angular coordinates
    theta = torch.linspace(0, 2*np.pi, n_theta)
    # Time coordinates
    t = torch.linspace(t_range[0], t_range[1], n_t)

    # Create meshgrid
    R, THETA, T = torch.meshgrid(r, theta, t, indexing='ij')

    # Convert to Cartesian coordinates
    X = R * torch.cos(THETA)
    Y = R * torch.sin(THETA)

    # Reshape to 2D tensors
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)

    return X_flat, Y_flat, T_flat

def generate_circle_boundary(n_theta=50, n_t=10, radius=1.0, t_range=(0, 1)):
    """Generate points on the circular boundary."""
    # Angular coordinates for boundary
    theta = torch.linspace(0, 2*np.pi, n_theta)
    # Time coordinates
    t = torch.linspace(t_range[0], t_range[1], n_t)

    # Create meshgrid
    THETA, T = torch.meshgrid(theta, t, indexing='ij')

    # Convert to Cartesian coordinates (on circle)
    X = radius * torch.cos(THETA)
    Y = radius * torch.sin(THETA)

    # Reshape to 2D tensors
    x_boundary = X.reshape(-1, 1)
    y_boundary = Y.reshape(-1, 1)
    t_boundary = T.reshape(-1, 1)

    return x_boundary, y_boundary, t_boundary

def gaussian_initial_condition(x, y, x0=0.0, y0=0.0, sigma=0.2):
    """Create an initial Gaussian temperature distribution."""
    r = torch.sqrt((x - x0)**2 + (y - y0)**2)
    return torch.exp(-r**2 / (2 * sigma**2))

def train_pinn(model, num_epochs=2000, learning_rate=0.001):
    """Train the PINN model."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    # Generate interior training points in circle
    x_train, y_train, t_train = generate_circle_domain()

    # Generate boundary points on circle
    x_boundary, y_boundary, t_boundary = generate_circle_boundary()

    # Set boundary condition (fixed temperature at boundary, e.g., u=0)
    u_boundary = torch.zeros_like(x_boundary)

    # Generate initial condition points
    radius = 1.0
    n_points = 50
    x = torch.linspace(-radius, radius, n_points)
    y = torch.linspace(-radius, radius, n_points)
    X_init, Y_init = torch.meshgrid(x, y, indexing='ij')

    # Create mask for points inside the circle
    mask = X_init**2 + Y_init**2 <= radius**2

    # Filter points to keep only those inside the circle
    x_initial = X_init[mask].reshape(-1, 1)
    y_initial = Y_init[mask].reshape(-1, 1)
    t_initial = torch.zeros_like(x_initial)

    # Apply initial condition
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

def plot_circular_solution(model, t_value=0.5, radius=1.0):
    """Plot the solution in a circular domain at a specific time."""
    n = 100
    x = torch.linspace(-radius, radius, n)
    y = torch.linspace(-radius, radius, n)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Create mask for points inside the circle
    mask = X**2 + Y**2 <= radius**2

    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    t_flat = torch.ones_like(x_flat) * t_value

    with torch.no_grad():
        u_pred = model(x_flat, y_flat, t_flat)
        U = u_pred.reshape(n, n)

    # Apply mask (set points outside circle to NaN)
    U_masked = U.clone()
    U_masked[~mask] = float('nan')

    plt.figure(figsize=(10, 8))
    plt.contourf(X.numpy(), Y.numpy(), U_masked.numpy(), cmap=cm.viridis, levels=50)
    plt.colorbar(label='Temperature')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Temperature at t = {t_value}')
    plt.axis('equal')
    circle = plt.Circle((0, 0), radius, fill=False, color='k')
    plt.gca().add_patch(circle)
    plt.tight_layout()
    plt.show()

# Add a main function to run the model
def main():
    # Create and train the model
    model = HeatEquation2DPINN()
    loss_history = train_pinn(model)

    # Plot the loss history
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.grid(True)
    plt.show()

    # Plot solutions at different time points
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        plot_circular_solution(model, t_value=t)

if __name__ == "__main__":
    main()
