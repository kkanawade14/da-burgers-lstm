import numpy as np
import matplotlib.pyplot as plt

# ============================
# Parameters for Both Solvers
# ============================
nx = 1000  # Number of grid points
nt = 500  # Number of time steps
dt = 0.001  # Initial time step size
mu = 0.01  # Viscosity (diffusion coefficient)

xlen = 2 * np.pi  # Domain length
dx = xlen / (nx - 1)
xcords = np.linspace(0, xlen, nx, endpoint=False)  # Periodic grid
t_values = np.linspace(0, nt * dt, nt)  # Time grid

c1 = 3  # Computed wave speed based on initial condition

# CFL Conditions
CFL_conv = 0.5  # Stability limit for convective term
CFL_diff = 0.1  # Stability limit for diffusive term

# Compute Maximum Allowed Time Step
dt_conv = CFL_conv * dx / c1
dt_diff = CFL_diff * dx**2 / mu

# Choose the most restrictive time step
dt_stable = min(dt_conv, dt_diff)

# Adjust dt if necessary
if dt > dt_stable:
    print(f"Warning: dt={dt} is too large! Adjusting to dt={dt_stable:.6e} for stability.")
    dt = dt_stable

print(f"Using dt = {dt:.6e} (CFL-constrained)")


t_values = np.linspace(0, nt * dt, nt)  # Time grid


# Storage for numerical and analytical solutions
numerical_solutions = np.zeros((nt, nx))
analytical_solutions = np.zeros((nt, nx))


# ============================
# Initial Condition
# ============================
def initial_condition(u_x):
    """Define the initial condition for the numerical solver."""
    for i in range(nx):
        if 0 <= xcords[i] <= np.pi:
            u_x[i] = xcords[i] + c1
        else:
            u_x[i] = 3 * xcords[i] / np.pi - c1 - 0.07
    return u_x


# ============================
# Burgers' Equation RHS (Numerical Solver)
# ============================
# Right-hand side function for Burgers' equation
def rhs_burgers(u, mu, dx):
    
    rhs = np.zeros_like(u)
    rhs[1:-1] = (-u[1:-1] * (u[1:-1] - u[:-2]) / dx  # Advection term
                 + mu * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2)  # Diffusion term
    
    # Periodic boundary conditions
    rhs[0] = (-u[0] * (u[0] - u[-2]) / dx
              + mu * (u[1] - 2 * u[0] + u[-2]) / dx**2)

    rhs[-1] = rhs[0]  # Periodicity
    return rhs



# ============================
# RK4 Time Integration
# ============================
def rk4_step(u, dt):
    """Fourth-order Runge-Kutta step for Burgers' equation."""
    k1 = rhs_burgers(u, mu, dx)
    k2 = rhs_burgers(u + 0.5 * dt * k1, mu, dx)
    k3 = rhs_burgers(u + 0.5 * dt * k2, mu, dx)
    k4 = rhs_burgers(u + dt * k3, mu, dx)
    return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# ============================
# Analytical Solution
# ============================
def analytical_solution(x, t, nu, c1):

    lambda_factor = nu / (xlen**2)  # Damping effect due to viscosity
    c_eff = c1 * np.exp(-1*lambda_factor * t)

    """Computes the analytical solution for the viscous Burgers' equation."""
    x = np.mod(x - c1 * t, xlen)  # Shift with wave speed

    phi_x = (-2 * (x) / (4 * nu * (t + 1)) * np.exp(-x ** 2 / (4 * nu * (t + 1))) +
             -2 * (x - xlen) / (4 * nu * (t + 1)) * np.exp(-(x - xlen) ** 2 / (4 * nu * (t + 1))))

    phi = (np.exp(-x ** 2 / (4 * nu * (t + 1))) +
           np.exp(-(x - xlen) ** 2 / (4 * nu * (t + 1))))

    return -2 * nu * phi_x / phi + c_eff


# ============================
# Run the Simulations
# ============================

# Initialize numerical solution
u_x = initial_condition(np.zeros(nx))
numerical_solutions[0, :] = u_x

# Run the numerical solver
for t in range(1, nt):
    u_x = rk4_step(u_x, dt)
    numerical_solutions[t, :] = u_x

# Compute the analytical solution at each time step
for t_idx in range(nt):
    t = t_values[t_idx]
    analytical_solutions[t_idx, :] = analytical_solution(xcords, t, mu, c1)


# ============================
# Animated Evolution Plot
# ============================

fig, ax = plt.subplots(figsize=(8, 5))
line_num, = ax.plot(xcords, numerical_solutions[0, :], 'b-', lw=2, label="Numerical")
line_ana, = ax.plot(xcords, analytical_solutions[0, :], 'r--', lw=2, label="Analytical")
ax.set_xlim(0, xlen)
ax.set_ylim(np.min(numerical_solutions) - 1, np.max(numerical_solutions) + 1)
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.set_title("Evolution of Numerical vs. Analytical Solution")
ax.legend()

for t_idx in range(nt):
    line_num.set_ydata(numerical_solutions[t_idx, :])  # Update numerical wave
    line_ana.set_ydata(analytical_solutions[t_idx, :])  # Update analytical wave
    plt.pause(0.01)  # Pause for animation effect

plt.show()




# ============================
# Plot Results
# ============================

# --- Side-by-side heatmaps ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(numerical_solutions, aspect='auto', cmap='jet', origin='lower')
axes[0].set_title("Numerical Solution")
axes[0].set_xlabel("Spatial Index (x)")
axes[0].set_ylabel("Time Index (t)")

axes[1].imshow(analytical_solutions, aspect='auto', cmap='jet', origin='lower')
axes[1].set_title("Analytical Solution")
axes[1].set_xlabel("Spatial Index (x)")
axes[1].set_ylabel("Time Index (t)")

plt.colorbar(axes[1].imshow(analytical_solutions, aspect='auto', cmap='jet', origin='lower'),
             ax=axes, label="u(x, t)")
plt.show()


# --- Overlay numerical and analytical at t=0, 50, 90 ---
time_steps = [0 ]

plt.figure(figsize=(8, 5))
for t_idx in time_steps:
    t = t_values[t_idx]
    plt.plot(xcords, analytical_solutions[t_idx, :], linestyle="--", label=f"Analytical t={t:.2f}", alpha=0.8)
    plt.plot(xcords, numerical_solutions[t_idx, :], linestyle="-", label=f"Numerical t={t:.2f}")

plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.title("Comparison at t=0, 50, 90")
plt.legend()
plt.grid()
plt.show()


