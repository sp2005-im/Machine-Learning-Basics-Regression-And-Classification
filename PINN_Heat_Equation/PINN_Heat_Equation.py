#Sri Rama Jayam
#PINN for the Heat Equation
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class HeatEquationPINN(nn.Module):
  def __init__(self, num_layer = 4, hidden_size = 20):
    super(HeatEquationPINN,self).__init__()
    #layers of the neural network
    layers = []
    # Input layer (x,t) -> 2 input features
    layers.append(nn.Linear(2,hidden_size))
    # Using the tanh (hyperbolic tan) as the activation function
    layers.append(nn.Tanh())

    #Hidden layers
    for _ in range(num_layer-1):
      layers.append(nn.Linear(hidden_size, hidden_size))
      layers.append(nn.Tanh())
    
    #Output layer
    layers.append(nn.Linear(hidden_size,1))

    self.net = nn.Sequential(*layers)

    #Heat equation parameter (thermal diffusivity)
    self.alpha = 0.1
  
  def forward(self,x,t):
    inputs = torch.cat([x,t], dim = 1)
    return self.net(inputs)
  
  def compute_derivatives(self,x,t):
    #Computing the derivatives for the heat equation
    x = x.requires_grad_(True)
    t = t.requires_grad_(True) # requires_grad_() -> in Torch, True if derivatives need to be computed, False otherwise

    #Forward pass
    u = self.forward(x,t)

    #First Derivatives
    du_dx = torch.autograd.grad(
        u,x,
        grad_outputs = torch.ones_like(u),
        create_graph=True #More useful for higher derivatives which we need
    )[0]

    du_dt = torch.autograd.grad(
        u,t,
        grad_outputs = torch.ones_like(u),
        create_graph = True
    )[0]

    du2_dx2 = torch.autograd.grad(
        du_dx,x,
        grad_outputs = torch.ones_like(du_dx),
        create_graph=True
    )[0]
    return du_dt, du2_dx2
  
  def pde_loss(self, x, t):
    # Function to compute PDE loss
    # Get Derivatives
    du_dt, du2_dx2 = self.compute_derivatives(x,t)
    pde_residual = du_dt - self.alpha*du2_dx2
    return torch.mean(pde_residual**2)
  
  def boundary_loss(self, x_boundary, t_boundary, u_boundary):
    #Computing the boundary condition loss
    u_pred = self.forward(x_boundary, t_boundary)
    return torch.mean((u_pred-u_boundary)**2)

  def initial_loss(self, x_initial, t_initial, u_initial):
    # Computing the initial condition loss
    u_pred = self.forward(x_initial, t_initial)
    return torch.mean((u_pred-u_initial)**2)
  

def generate_training_data(nx = 50, nt = 50, x_range = (0,1), t_range = (0,1)):
  #Generate training data points in the domain
  x = torch.linspace(x_range[0], x_range[1], nx)
  t = torch.linspace(t_range[0], t_range[1], nt)

  # Create meshgrid
  X, T = torch.meshgrid(x, t, indexing = 'ij')

  #Flatten to 2D tensors
  X_flat = X.reshape(-1,1) #-1 -> To convert to the right shape (automatically), 1 -> To flatten to 1 column
  T_flat = T.reshape(-1,1)
  return X_flat, T_flat 

def train_pinn(model, num_epochs = 1000, learning_rate = 0.001):
  # Training the PINN model
  optimizer = optim.Adam(model.parameters(), lr = learning_rate)
  x_train, t_train = generate_training_data()

  #Example training data
  def initial_condition(x):
    return torch.sin(np.pi*x)
  
  loss_history = []

  #Training loop 
  for epoch in range(num_epochs):
    optimizer.zero_grad()

    #Compute PDE loss
    pde_loss = model.pde_loss(x_train, t_train)

    #Compute boundary losses
    x_boundary = torch.tensor([[0.0], [1.0]]).float()
    t_boundary = torch.zeros_like(x_boundary)
    u_boundary = torch.zeros_like(x_boundary) 
    bc_loss = model.boundary_loss(x_boundary, t_boundary, u_boundary)

    #Compute initial losses
    x_initial = torch.linspace(0,1,100).reshape(-1,1) 
    t_initial = torch.zeros_like(x_initial)
    u_initial = initial_condition(x_initial)
    ic_loss = model.initial_loss(x_initial, t_initial, u_initial)

    #Compute the total loss
    total_loss = pde_loss + bc_loss + ic_loss

    #Backpropagation
    total_loss.backward()
    optimizer.step()
    
    loss_history.append(total_loss)

    if epoch%100 == 0:
      print(f'Epoch {epoch}, Loss: {total_loss.item():.6f}')
  return loss_history

if __name__ == "__main__":
  model = HeatEquationPINN()
  loss_history = train_pinn(model) 
  with torch.no_grad():
    x_test, t_test = generate_training_data(nx=100, nt=100)
    u_pred = model(x_test, t_test)
        
    # Reshape for plotting
    X = x_test.reshape(100, 100)
    T = t_test.reshape(100, 100)
    U = u_pred.reshape(100, 100)
    #print(len(loss_history))
    
    plt.figure(figsize=(10, 8))
    plt.contourf(X, T, U)
    plt.colorbar(label='Temperature')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Heat Equation Solution using PINN')
    plt.show()
    
