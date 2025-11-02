# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# nu = 0.1  # Viscosity
# L = 2 * np.pi  # Periodicity length
# Nx = 100  # Number of spatial points
# Nt = 100  # Number of time steps
# T = 2.0   # Final time
# dt = T / Nt  # Time step size
# x = np.linspace(0, L, Nx)  # Spatial grid
# t_values = np.linspace(0, T, Nt)  # Time grid

# # Function for the analytical solution
# def analytical_solution(x, t, nu):
    
#     phi_x = (-2 * (x - 4 * t) / (4 * nu * (t + 1)) * np.exp(-(x - 4 * t) ** 2 / (4 * nu * (t + 1))) +
#              -2 * (x - 4 * t - 2 * np.pi) / (4 * nu * (t + 1)) * np.exp(-(x - 4 * t - 2 * np.pi) ** 2 / (4 * nu * (t + 1))))

#     phi = (np.exp(-(x - 4 * t) ** 2 / (4 * nu * (t + 1))) + 
#            np.exp(-(x - 4 * t - 2 * np.pi) ** 2 / (4 * nu * (t + 1))))
    
#     return -2 * nu * phi_x / phi + 4 

# # Initialize solution storage and figure
# solutions = np.zeros((Nt, Nx))
# fig, ax = plt.subplots()
# line, = ax.plot(x, np.zeros_like(x), 'b-', lw=2)
# ax.set_xlim(0, L)
# ax.set_ylim(-7, 7)  # Adjusted for expected range
# ax.set_xlabel("x")
# ax.set_ylabel("u(x, t)")
# ax.set_title("Evolving Solution of 1D Burgers' Equation")

# # Time evolution loop with animation
# for t_idx in range(Nt):
#     t = t_values[t_idx]
#     u_x = analytical_solution(x, t, nu)
#     solutions[t_idx, :] = u_x  # Store the solution
#     line.set_ydata(u_x)  # Update the plot
#     plt.pause(0.01)  # Pause for animation effect

# # Show the final result
# plt.title("Final Solution")
# plt.imshow(solutions)
# plt.show()




import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.01  # Viscosity
L = 2 * np.pi  # Periodicity length
Nx = 100  # Number of spatial points
Nt = 100  # Number of time steps
T = 1.0   # Final time
dt = T / Nt  # Time step size
x = np.linspace(0, L, Nx, endpoint=False)  # Ensure periodic grid without duplicate point
t_values = np.linspace(0, T, Nt)  # Time grid

c1 = 4
# Function for the analytical solution
def analytical_solution(x, t, nu,c1):
    x = np.mod(x - c1 * t, L)  # Enforce periodicity in the calculation

    phi_x = (-2 * (x) / (4 * nu * (t + 1)) * np.exp(-x ** 2 / (4 * nu * (t + 1))) +
             -2 * (x - L) / (4 * nu * (t + 1)) * np.exp(-(x - L) ** 2 / (4 * nu * (t + 1))))

    phi = (np.exp(-x ** 2 / (4 * nu * (t + 1))) + 
           np.exp(-(x - L) ** 2 / (4 * nu * (t + 1))))
    
    return -2 * nu * phi_x / phi + c1 


#print
u0 = analytical_solution(x, 0, nu,c1)

print(u0[0])
print(u0[-1])
print(u0[49])
print(u0[50])
print(u0[51])
print(u0[52])
print(u0[53])


plt.figure
plt.plot(x,u0)
plt.show()

# Initialize solution storage and figure
solutions = np.zeros((Nt, Nx))
fig, ax = plt.subplots()
line, = ax.plot(x, np.zeros_like(x), 'b-', lw=2)
ax.set_xlim(0, L)
ax.set_ylim(-7, 7)  # Adjusted for expected range
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.set_title("Evolving Solution of 1D Burgers' Equation with Periodicity")

# Time evolution loop with animation
for t_idx in range(Nt):
    t = t_values[t_idx]
    u_x = analytical_solution(x, t, nu,c1)
    solutions[t_idx, :] = u_x  # Store the solution
    line.set_ydata(u_x)  # Update the plot
    plt.pause(0.01)  # Pause for animation effect


plt.figure()
# Show the final result
plt.title("Final Solution with Periodicity")
plt.imshow(solutions, aspect='auto', cmap='jet', origin='lower')
plt.colorbar(label="u(x, t)")
plt.show()


