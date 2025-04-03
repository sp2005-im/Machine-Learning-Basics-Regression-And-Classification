import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm

class ConvectionEquationPINN(nn.Module):
    def __init__(self, num_layer=4, hidden_size=20, c=1.0):
        super(ConvectionEquationPINN, self).__init__()
        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_layer - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)
        self.c = c  # Convection speed

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

    def compute_derivatives(self, x, t):
        x.requires_grad = True
        t.requires_grad = True
        
        inputs = torch.cat([x, t], dim=1)
        u = self.net(inputs)

        du_dx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
        )[0]
        
        du_dt = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        
        return du_dt, du_dx

    def pde_loss(self, x, t):
        du_dt, du_dx = self.compute_derivatives(x, t)
        pde_residual = du_dt + self.c * du_dx  # Convection equation: du/dt + c*du/dx = 0
        return torch.mean(pde_residual**2)

    def initial_loss(self, x_initial, t_initial, u_initial):
        u_pred = self.forward(x_initial, t_initial)
        return torch.mean((u_pred - u_initial) ** 2)
    
    def boundary_loss(self, x_left, t_left, x_right, t_right):
        # Periodic boundary conditions: u(x=0, t) = u(x=2, t)
        u_left = self.forward(x_left, t_left)
        u_right = self.forward(x_right, t_right)
        return torch.mean((u_left - u_right) ** 2)

def generate_training_data(nx=50, nt=50, x_range=(0, 2), t_range=(0, 1)):
    x = torch.linspace(x_range[0], x_range[1], nx)
    t = torch.linspace(t_range[0], t_range[1], nt)
    X, T = torch.meshgrid(x, t, indexing='ij')
    return X.reshape(-1, 1), T.reshape(-1, 1)

def initial_condition(x):
    u = torch.ones_like(x)
    mask = (x.flatten() > 0.5) & (x.flatten() < 1)
    u[mask] = 2
    return u.reshape(-1, 1)

def analytical_solution(x, t, c=1.0):
    """
    Analytical solution for the convection equation with the given initial condition.
    For the square pulse, it just shifts to the right with speed c.
    """
    # Ensure x and t are numpy arrays
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().numpy()
        
    x_shifted = x - c * t
    
    # Apply periodicity for x out of bounds
    x_shifted = x_shifted % 2.0
    
    # Initialize with ones
    u = np.ones_like(x_shifted)
    
    # Set the pulse
    mask = (x_shifted > 0.5) & (x_shifted < 1.0)
    u[mask] = 2.0
    
    return u

def train_pinn(model, num_epochs=2000, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    x_train, t_train = generate_training_data(nx=100, nt=100)

    # Data for initial condition
    x_initial = torch.linspace(0, 2, 200).reshape(-1, 1)
    t_initial = torch.zeros_like(x_initial)
    u_initial = initial_condition(x_initial)
    
    # Data for boundary conditions
    t_boundary = torch.linspace(0, 1, 100).reshape(-1, 1)
    x_left = torch.zeros_like(t_boundary)
    x_right = torch.ones_like(t_boundary) * 2.0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5, verbose=True)
    loss_history = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # PDE residual loss
        pde_loss = model.pde_loss(x_train, t_train)
        
        # Initial condition loss
        ic_loss = model.initial_loss(x_initial, t_initial, u_initial)
        
        # Boundary condition loss (periodic)
        bc_loss = model.boundary_loss(x_left, t_boundary, x_right, t_boundary)

        # Total loss with weighting
        total_loss = pde_loss + 10.0 * ic_loss + 10.0 * bc_loss
        
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item():.6f}, PDE: {pde_loss.item():.6f}, IC: {ic_loss.item():.6f}, BC: {bc_loss.item():.6f}')
            scheduler.step(total_loss)
            
    return loss_history

def evaluate_model(model, nx=100, nt=20, x_range=(0, 2), t_range=(0, 1)):
    # Create a grid for evaluation
    x = torch.linspace(x_range[0], x_range[1], nx).reshape(-1, 1)
    t = torch.linspace(t_range[0], t_range[1], nt)
    
    u_pred = np.zeros((nx, nt))
    u_analytical = np.zeros((nx, nt))
    
    # Evaluate the model at each time step
    for i, t_val in enumerate(t):
        t_tensor = torch.ones_like(x) * t_val
        u_pred[:, i] = model(x, t_tensor).detach().numpy().flatten()
        u_analytical[:, i] = analytical_solution(x.numpy(), t_val, c=model.c).flatten()
    
    return x.numpy(), t.numpy(), u_pred, u_analytical

def plot_results(model, loss_history):
    plt.figure(figsize=(12, 4))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.tight_layout()
    plt.savefig('pinn_loss.png')
    plt.show()
    
    # Evaluate the model
    x, t, u_pred, u_analytical = evaluate_model(model)
    X, T = np.meshgrid(x.flatten(), t, indexing='ij')
    
    # Plot predicted solution
    fig = plt.figure(figsize=(18, 6))
    
    # PINN prediction
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, T, u_pred, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u')
    ax1.set_title('PINN Solution')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # Analytical solution
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, T, u_analytical, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u')
    ax2.set_title('Analytical Solution')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # Error
    ax3 = fig.add_subplot(133, projection='3d')
    error = np.abs(u_pred - u_analytical)
    surf3 = ax3.plot_surface(X, T, error, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    ax3.set_zlabel('Error')
    ax3.set_title('Absolute Error')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig('pinn_solution.png')
    plt.show()
    
    # Plot solution at different time steps
    time_steps = [0, int(len(t)/4), int(len(t)/2), int(3*len(t)/4), -1]
    plt.figure(figsize=(15, 10))
    
    for i, step in enumerate(time_steps):
        plt.subplot(len(time_steps), 1, i+1)
        plt.plot(x, u_pred[:, step], 'b-', label='PINN Prediction')
        plt.plot(x, u_analytical[:, step], 'r--', label='Analytical')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title(f't = {t[step]:.2f}')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('pinn_time_slices.png')
    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create the model
    model = ConvectionEquationPINN(num_layer=5, hidden_size=40, c=1.0)
    
    # Train the model
    loss_history = train_pinn(model, num_epochs=2000, learning_rate=0.001)
    
    # Plot results
    plot_results(model, loss_history)

if __name__ == "__main__":
    main()
