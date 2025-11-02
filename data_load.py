import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


def solve_one_step(u, mu, xlen, nx , dt):
    # Right-hand side function for Burgers' equation
    u = u
    mu = mu
    dx = xlen / (nx - 1)
    dt = dt

    
    def rhs_burgers(u, mu, dx):
        
        rhs = np.zeros_like(u)
        rhs[1:-1] = (-u[1:-1] * (u[1:-1] - u[:-2]) / dx  # Advection term
                    + mu * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2)  # Diffusion term
        
        # Periodic boundary conditions
        rhs[0] = (-u[0] * (u[0] - u[-2]) / dx
                + mu * (u[1] - 2 * u[0] + u[-2]) / dx**2)
        rhs[-1] = rhs[0]  # Periodicity
        return rhs


    # RK4 Method for time integration
    def rk4_step(u, dt):
        k1 = rhs_burgers(u, mu, dx)
        k2 = rhs_burgers(u + 0.5 * dt * k1, mu, dx)
        k3 = rhs_burgers(u + 0.5 * dt * k2, mu, dx)
        k4 = rhs_burgers(u + dt * k3, mu, dx)
        return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    
    u_next = rk4_step(u, dt)
    return np.transpose(u_next)


def solve_multi_step(u, mu, xlen, nx ,dt, steps):
    # Right-hand side function for Burgers' equation
    u = u
    mu = mu
    dx = xlen/( nx - 1)
    dt = dt
    steps = steps

    solutions = np.zeros((nx,steps))

    def rhs_burgers(u, mu, dx):
        
        rhs = np.zeros_like(u)
        rhs[1:-1] = (-u[1:-1] * (u[1:-1] - u[:-2]) / dx  # Advection term
                    + mu * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2)  # Diffusion term
        
        # Periodic boundary conditions
        rhs[0] = (-u[0] * (u[0] - u[-2]) / dx
                + mu * (u[1] - 2 * u[0] + u[-2]) / dx**2)
        rhs[-1] = rhs[0]  # Periodicity
        return rhs


    # RK4 Method for time integration
    def rk4_step(u, dt):
        k1 = rhs_burgers(u, mu, dx)
        k2 = rhs_burgers(u + 0.5 * dt * k1, mu, dx)
        k3 = rhs_burgers(u + 0.5 * dt * k2, mu, dx)
        k4 = rhs_burgers(u + dt * k3, mu, dx)
        return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    
    
    
    for t in range(steps):
        u_next = rk4_step(u, dt)
        solutions[:, t] = np.transpose(u_next)
        u = u_next
    

    return solutions


def add_noise_to_truth(solution, noise_std):
    noise = np.random.normal(0, noise_std, solution.shape)
    noisy_solution = solution + noise
    return noisy_solution


def solve1dburgers(xlen, nx, nt, dt, mu, delta):
# Parameters for the Burgers' equation
    # nx = 101        # Number of grid points
    # nt = 100        # Number of time steps
    # dt = 0.001      # Time step size
    # mu = 0.01       # Viscosity (diffusion coefficient)

    
    #xlen = 1        # Length of the domain
    dx = xlen / (nx - 1)  
    xcords = np.linspace(0, xlen, nx)  
    u_x = np.zeros(nx)  
    solutions = np.zeros((nx,nt+1))

    # Define the initial condition
    def initial_condition(u_x):
        
        for i in range(nx):
            if 0 <= xcords[i] <= np.pi:  # Square wave initial condition
                u_x[i] = xcords[i] + 3 
            else:
                u_x[i] = 3*xcords[i]/np.pi - 3
        return u_x


    # Right-hand side function for Burgers' equation
    def rhs_burgers(u):
        
        rhs = np.zeros_like(u)
        rhs[1:-1] = (-u[1:-1] * (u[1:-1] - u[:-2]) / dx  # Advection term
                    + mu * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2)  # Diffusion term
        
        # Periodic boundary conditions
        rhs[0] = (-u[0] * (u[0] - u[-2]) / dx
                + mu * (u[1] - 2 * u[0] + u[-2]) / dx**2)
        rhs[-1] = rhs[0]  # Periodicity
        return rhs


    # RK4 Method for time integration
    def rk4_step(u, dt):
        k1 = rhs_burgers(u)
        k2 = rhs_burgers(u + 0.5 * dt * k1)
        k3 = rhs_burgers(u + 0.5 * dt * k2)
        k4 = rhs_burgers(u + dt * k3)
        return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Main loop
    
    u_x = initial_condition(u_x)
    noise = np.random.normal(0, delta, u_x.shape)
    u_x += noise  
    solutions[:,0] = np.transpose(u_x)  # Store the initial condition at time 0

    for t in range(nt):
        u_x = rk4_step(u_x, dt)
        solutions[:,t+1] = np.transpose(u_x)

    return solutions


def analytical1dburgers(c,xlen, nx, nt, dt, mu):
    
    nu = mu  # Viscosity
    L = xlen  # Periodicity length
    Nx = nx  # Number of spatial points
    Nt = nt  # Number of time steps
    
    
    T = dt * nt  # Time step size
    x = np.linspace(0, L, nx)  # Ensure periodic grid without duplicate point
    t_values = np.linspace(0, T, nt)  # Time grid

    c1 = c
    # Function for the analytical solution
    def analytical_solution(x, t, nu,c1):
        x = np.mod(x - c1 * t, L)  # Enforce periodicity in the calculation

        phi_x = (-2 * (x) / (4 * nu * (t + 1)) * np.exp(-x ** 2 / (4 * nu * (t + 1))) +
                -2 * (x - L) / (4 * nu * (t + 1)) * np.exp(-(x - L) ** 2 / (4 * nu * (t + 1))))

        phi = (np.exp(-x ** 2 / (4 * nu * (t + 1))) + 
            np.exp(-(x - L) ** 2 / (4 * nu * (t + 1))))
        
        return -2 * nu * phi_x / phi + c1 


    # Initialize solution storage and figure
    solutions = np.zeros((Nx, Nt))
    
    # Time evolution loop with animation
    for t_idx in range(Nt):
        t = t_values[t_idx]
        u_x = analytical_solution(x, t, nu,c1)
        solutions[:,t_idx] = u_x  # Store the solution

    return solutions    


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BurgersDataset(Dataset):
    def __init__(self, c,xlen, nx, nt, dt, mu, seq_length, noise_std=0.00):
        """
        Args:
            xlen (float): Length of the domain
            nx (int): Number of spatial points
            nt (int): Number of time steps
            dt (float): Time step size
            mu (float): Viscosity parameter
            seq_length (int): Number of previous time steps used for prediction
            noise_std (float): Standard deviation of Gaussian noise added to data
        """
        self.data = analytical1dburgers(c,xlen, nx, nt, dt, mu)  # Generate data
        self.nx, self.nt = self.data.shape
        self.seq_length = seq_length
        
        # # Normalize data (optional)
        # self.mean = np.mean(self.data)
        # self.std = np.std(self.data)
        # self.data = (self.data - self.mean) / self.std

        # Add noise if required
        if noise_std > 0:
            self.data += np.random.normal(0, noise_std, self.data.shape)
        
    def __len__(self):
        return self.nt - self.seq_length  # Ensuring we start predicting from 11th timestep
    
    def __getitem__(self, idx):
        """Returns a sequence of past states and the next state"""
        x = self.data[:,idx:idx + self.seq_length].T  # Shape: (seq_length, nx)
        y = self.data[:,idx + self.seq_length].T      # Shape: (nx,)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_dataloader(c,xlen, nx, nt, dt, mu, seq_length, batch_size=32, shuffle=False, noise_std=0.01):

    xlen = xlen
    nx=nx
    nt=nt
    dt=dt
    mu=mu
    c1=c
    seq_length=seq_length
    noise_std=noise_std
    
    if nt < 10:
        print("Dataset can not be created. Need more than 10 timesteps")
    else:
        dataset = BurgersDataset(c1,xlen, nx, nt, dt, mu, seq_length, noise_std)
    
    return DataLoader(dataset, batch_size=batch_size)


def burgers_jacobian(u, dx, dt, nu):
    N = len(u)
    u = u.flatten()
    # print(f"N is {N}")
    # print(f"u is {u.shape}")
    Dx = np.zeros((N, N))
    Dxx = np.zeros((N, N))
    for i in range(N):
        Dx[i, (i - 1) % N] = -0.5 / dx
        Dx[i, (i + 1) % N] =  0.5 / dx
        Dxx[i, (i - 1) % N] =  1.0 / dx**2
        Dxx[i, i]           = -2.0 / dx**2
        Dxx[i, (i + 1) % N] =  1.0 / dx**2

    U = np.diag(u)
    dudx = Dx @ u
    D_dudx = np.diag(dudx)
    # print(f"Shape of U is {U.shape}")
    # print(f"Shape of Dx is {Dx.shape}")
    # print(f"Shape of Dx is {Dxx.shape}")
    term1 = dt * nu * Dxx
    term2 = dt * (D_dudx + U @ Dx)

    A = np.eye(N) - term2 + term1
    
    
    return A




