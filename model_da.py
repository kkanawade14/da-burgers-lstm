import numpy as np
import torch
from data_load import solve1dburgers, solve_one_step, burgers_jacobian
from model import LSTMBurgers
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os

hidden_size=256
num_layers=6

def load_trained_model(model_path, input_size, device="cpu",hidden_size=hidden_size, num_layers=num_layers):
    
    model = LSTMBurgers(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def prepare_input(data, idx, seq_length, device="cpu"):
    if idx < seq_length:
        raise ValueError("not enough sequence of past states.")
    
    x = data[:, idx - seq_length:idx].T  
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  


def BLUE_DA(ground_truth, observations, r_values, model, device=None,
                          seq_length=5, xlen=2.0, nx=100, dt=0.01, mu=0.01, nt_total=50, delta=1e-3):
    
    all_analysis_errors = np.zeros((len(r_values), nt_total-2))
    
    
    for idx, r in enumerate(r_values):
        print(f"Processing r = {r} in BLUE ({idx+1}/{len(r_values)})")
        
       
        analysis_errors = []
        
        
        sigma_o = 0.01
        sigma_b = np.sqrt(r) * sigma_o
        
        
        W_mat = (r / (r + 1)) * np.eye(nx)
        
        
        X1_b_sol = solve1dburgers(xlen, nx, 1, dt, mu, delta)    
        X_b = X1_b_sol[:, [1]]
        
       
        X_a = X_b + W_mat @ (observations[:, [1]] - X_b)
        
        
        error = np.sqrt(np.mean((X_a - ground_truth[:, [1]])**2))
        analysis_errors.append(error)
        
        
        X1_b_sol[:, [1]] = X_a
        
        
        for step in range(nt_total-3):
            current_step = step + 2  
            
            
            if current_step >= seq_length:  
                
                input_tensor = prepare_input(X1_b_sol, current_step, seq_length, device)
                
                with torch.no_grad():
                    u_pred = model(input_tensor)
                
                X_b = u_pred.reshape(nx, 1).cpu().numpy()
                
            else:
                
                x_b_array = solve_one_step(X_a.flatten(), mu, xlen, nx, dt)
                X_b = x_b_array.reshape(nx, 1)
            
            
            X_a = X_b + W_mat @ (observations[:, [current_step]] - X_b)
            
            
            error = np.sqrt(np.mean((X_a - ground_truth[:, [current_step]])**2))
            analysis_errors.append(error)
            
            
            X1_b_sol = np.hstack((X1_b_sol, X_a))
        
        
        all_analysis_errors[idx, :] = analysis_errors
    
    return all_analysis_errors

def EKF_DA(ground_truth, observations, r_values, model, device=None,
                         seq_length=5, xlen=2.0, nx=100, dt=0.01, mu=0.01, nt_total=50, delta=1e-3):

    # Compute spatial step size
    dx = xlen / nx
    

    H = np.eye(nx)
    

    all_analysis_errors = np.zeros((len(r_values), nt_total-2))
    

    for idx, r in enumerate(r_values):
        print(f"Processing r = {r} in Extended Kalman Filter ({idx+1}/{len(r_values)})")
        

        analysis_errors = []
        

        sigma_o = 0.01
        sigma_b = np.sqrt(r) * sigma_o

        R = sigma_o**2 * np.eye(nx)
        

        Q = 0 * np.eye(nx)
        
        # Solve for initial background state
        X1_b_sol = solve1dburgers(xlen, nx, 1, dt, mu, delta)    
        u0 = X1_b_sol[:, [0]]
        X_b = X1_b_sol[:, [1]]
        P_k = sigma_b**2 * np.eye(nx)
        F_k = burgers_jacobian(u0, dx, dt, mu)
        P_f = F_k @ P_k @ F_k.T # last time when r in in 
        
        
        K = P_f @ H.T @ np.linalg.inv(H @ P_f @ H.T + R)
        
        
        X_a = X_b + K @ (observations[:, [1]] - H @ X_b)
        
        
        P_a = (np.eye(nx) - K @ H) @ P_f
        
   
        error = np.sqrt(np.mean((X_a - ground_truth[:, [1]])**2))
        analysis_errors.append(error)
        
    
        X1_b_sol[:, [1]] = X_a
        
       
        for step in range(nt_total-3):
            current_step = step + 2
            
           
            if current_step >= seq_length:  
                
                input_tensor = prepare_input(X1_b_sol, current_step, seq_length, device)
                
                
                with torch.no_grad():
                    u_pred = model(input_tensor)
                
                
                X_b = u_pred.reshape(nx, 1).cpu().numpy()
                
            
                F_k = burgers_jacobian(X_a.flatten(), dx, dt, mu)
                
            
                P_f = F_k @ P_a @ F_k.T + Q
                
            else:
            
                x_b_array = solve_one_step(X_a.flatten(), mu, xlen, nx, dt)
                X_b = x_b_array.reshape(nx, 1)
                
            
                F_k = burgers_jacobian(X_a.flatten(), dx, dt, mu)
                
            
                P_f = F_k @ P_a @ F_k.T + Q
            
            
            K = P_f @ H.T @ np.linalg.inv(H @ P_f @ H.T + R)
            
            
            X_a = X_b + K @ (observations[:, [current_step]] - H @ X_b)
            
            
            P_a = (np.eye(nx) - K @ H) @ P_f
            
            
            error = np.sqrt(np.mean((X_a - ground_truth[:, [current_step]])**2))
            analysis_errors.append(error)
            
            
            X1_b_sol = np.hstack((X1_b_sol, X_a))
        
    
        all_analysis_errors[idx, :] = analysis_errors
    
    return all_analysis_errors

def create_localization_matrix(r, n):
    """
    Create a Gaussian localization matrix for covariance localization.
    
    Parameters:
    -----------
    r : int
        Localization radius
    n : int
        State dimension (number of grid points)
        
    Returns:
    --------
    L : numpy.ndarray
        Localization matrix
    """
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            # Account for periodicity in the domain
            dij = np.min([np.abs(i-j), np.abs((n-1)-j+i)])
            
            # Gaussian correlation function
            L[i, j] = (dij**2)/(2*r**2)
            L[j, i] = L[i, j]  # Ensure symmetry
    
    L = np.exp(-L)
    return L

def visualize_state_evolution(ground_truth, observations, analysis_state, r_value, nt_total, nx, save_path='./'):

    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Ground Truth
    im0 = axes[0].imshow(ground_truth[:, 1:nt_total], aspect='auto', origin='lower', 
                         extent=[0, nt_total-2, 0, nx], cmap='viridis')
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Spatial Index')
    plt.colorbar(im0, ax=axes[0])
    
    # Observations
    im1 = axes[1].imshow(observations[:, 1:nt_total], aspect='auto', origin='lower', 
                         extent=[0, nt_total-2, 0, nx], cmap='viridis')
    axes[1].set_title('Observations')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Spatial Index')
    plt.colorbar(im1, ax=axes[1])
    
    # Analysis
    im2 = axes[2].imshow(analysis_state, aspect='auto', origin='lower', 
                         extent=[0, nt_total-2, 0, nx], cmap='viridis')
    axes[2].set_title('EnKF Analysis')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Spatial Index')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/state_evolution_r{r_value}.png', dpi=300)
    plt.close()
    
    # Create animation of the time evolution (limited to 30 frames max)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    max_frames = min(nt_total-2, 30)
    
    # Plot initial state
    truth_line, = ax.plot(range(nx), ground_truth[:, 1], 'k-', label='Ground Truth')
    obs_line, = ax.plot(range(nx), observations[:, 1], 'b.', label='Observations')
    analysis_line, = ax.plot(range(nx), analysis_state[:, 0], 'r-', label='EnKF Analysis')
    
    ax.set_xlim(0, nx)
    y_min = min(ground_truth.min(), analysis_state.min())
    y_max = max(ground_truth.max(), analysis_state.max())
    margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin, y_max + margin)
    
    ax.set_xlabel('Spatial Index')
    ax.set_ylabel('State Value')
    ax.set_title(f'State Evolution (Time Step: 0, r = {r_value})')
    ax.legend()
    
    def update(frame):
        truth_line.set_ydata(ground_truth[:, frame+1])
        obs_line.set_ydata(observations[:, frame+1])
        analysis_line.set_ydata(analysis_state[:, frame])
        ax.set_title(f'State Evolution (Time Step: {frame}, r = {r_value})')
        return truth_line, obs_line, analysis_line
    
    anim = FuncAnimation(fig, update, frames=max_frames, interval=200, blit=True)
    anim.save(f'{save_path}/state_evolution_animation_r{r_value}.gif', writer='pillow', dpi=100)
    plt.close()
    
    # Plot specific time snapshots
    num_snapshots = min(4, nt_total-2)
    snapshot_times = np.linspace(0, nt_total-3, num_snapshots, dtype=int)
    
    fig, axes = plt.subplots(num_snapshots, 1, figsize=(10, 3*num_snapshots))
    if num_snapshots == 1:
        axes = [axes]
    
    for i, time_idx in enumerate(snapshot_times):
        axes[i].plot(range(nx), ground_truth[:, time_idx+1], 'k-', label='Ground Truth')
        axes[i].plot(range(nx), observations[:, time_idx+1], 'b.', label='Observations')
        axes[i].plot(range(nx), analysis_state[:, time_idx], 'r-', label='EnKF Analysis')
        axes[i].set_title(f'Time Step: {time_idx}')
        axes[i].set_xlabel('Spatial Index')
        axes[i].set_ylabel('State Value')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/state_snapshots_r{r_value}.png', dpi=300)
    plt.close()


def plot_analysis_variance(analysis_variance, r_values, dt, save_path='./'):
    
    # Convert time steps to actual time
    time_steps = np.arange(analysis_variance.shape[1]) * dt
    
    # Plot variance evolution for each r value
    plt.figure(figsize=(12, 8))
    
    markers = ['o', 's', '^', 'v', 'p', '*', 'h', 'H', 'D', 'd', '|', '_', 'x', '+', '.']
    num_styles = len(markers)
    cmap = plt.get_cmap('tab20b', len(r_values))
    
    for i, r in enumerate(r_values):
        marker = markers[i % num_styles]
        color = cmap(i)
        plt.plot(time_steps, analysis_variance[i, :], linestyle='-', marker=marker, 
                markevery=max(1, len(time_steps)//20), color=color, label=f'r = {r}')
    
    plt.xlabel('Time')
    plt.ylabel('Analysis Ensemble Variance')
    plt.title('Evolution of Analysis Uncertainty')
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.yscale('log')  # Often variance is better viewed on log scale
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    plt.savefig(f'{save_path}/analysis_variance_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot average variance for each r value
    avg_variance = np.mean(analysis_variance, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, avg_variance, 'b-o')
    plt.xlabel('r Value (Background/Observation Error Ratio)')
    plt.ylabel('Average Analysis Ensemble Variance')
    plt.title('Average Analysis Uncertainty vs. Error Ratio')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_path}/avg_analysis_variance_vs_r.png', dpi=300, bbox_inches='tight')
    plt.close()


# --- Modified EnKF_DA_RK4 to return prediction states as well ---
def EnKF_DA_RK4(ground_truth, observations, r_values, model, device=None,
                ensemble_size=20, ensemble_size_total=100, seq_length=5,
                xlen=2.0, nx=100, dt=0.01, mu=0.01,
                nt_total=50, delta=1e-3, localization_radius=3, inflation=1.02,
                save_path="./results"):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.random.seed(10)
    I = np.eye(nx)
    ones = np.ones(ensemble_size)
    ones_total = np.ones(ensemble_size_total)

    all_analysis_errors = np.zeros((len(r_values), nt_total - 2))
    all_forecast_errors = np.zeros((len(r_values), nt_total - 2))
    all_analysis_variance = np.zeros((len(r_values), nt_total - 2))
    all_predictions = np.zeros((len(r_values), nx, nt_total - 2))

    L = create_localization_matrix(localization_radius, nx)

    for idx, r in enumerate(r_values):
        print(f"Processing r = {r} in EnKF-RK4 ({idx+1}/{len(r_values)})")

        analysis_errors = []
        forecast_errors = []
        analysis_variances = []
        predictions = []

        sigma_o = 0.01
        ny = int(round(1 * nx))
        R_k = (sigma_o ** 2) * np.eye(ny)

        X1_b_sol = solve1dburgers(xlen, nx, 1, dt, mu, delta)
        xt_k = ground_truth[:, 1].copy()

        Xf_total = np.outer(ones_total, X1_b_sol[:, 1]) + 0.05 * np.random.randn(ensemble_size_total, nx)
        ind = np.random.permutation(ensemble_size_total)[:ensemble_size]
        Xf_k = Xf_total[ind, :].copy().T

        for k in range(nt_total - 2):
            current_step = k + 1
            xf_k = np.mean(Xf_k, axis=1)
            Pf_k = L * np.cov(Xf_k)

            forecast_error = np.linalg.norm(xf_k - xt_k)
            forecast_errors.append(forecast_error)

            obs_comp = np.random.permutation(ny)
            H_k = I[obs_comp, :]
            y_k = H_k @ observations[:, current_step]

            Eobs_k = sigma_o * np.random.randn(ny, ensemble_size)
            Yobs_k = np.outer(y_k, np.ones(ensemble_size)) + Eobs_k
            D_k = Yobs_k - H_k @ Xf_k
            IN_k = R_k + H_k @ Pf_k @ H_k.T
            Z_k = np.linalg.solve(IN_k, D_k)

            Xa_k = Xf_k + Pf_k @ H_k.T @ Z_k
            xa_k = np.mean(Xa_k, axis=1)
            DXa_k = Xa_k - np.outer(xa_k, ones)
            Xa_k = np.outer(xa_k, ones) + inflation * DXa_k

            analysis_error = np.linalg.norm(xa_k - xt_k)
            analysis_errors.append(analysis_error)

            analysis_variance = np.mean(np.var(Xa_k, axis=1))
            analysis_variances.append(analysis_variance)

            predictions.append(xa_k)

            X_a = xa_k.reshape(nx, 1)
            if k == 0:
                X1_b_sol[:, [1]] = X_a
            else:
                X1_b_sol = np.hstack((X1_b_sol, X_a))

            X_ensemble_forecast = np.zeros((nx, ensemble_size))
            for e in range(ensemble_size):
                current_state = Xa_k[:, e].reshape(nx, 1)
                x_forecast = solve_one_step(current_state.flatten(), mu, xlen, nx, dt)
                X_ensemble_forecast[:, e] = x_forecast

            Xf_k = X_ensemble_forecast
            xt_k = ground_truth[:, current_step + 1]

        all_analysis_errors[idx, :] = analysis_errors
        all_forecast_errors[idx, :] = forecast_errors
        all_analysis_variance[idx, :] = analysis_variances
        all_predictions[idx, :, :] = np.array(predictions).T

    return all_analysis_errors, all_forecast_errors, all_analysis_variance, all_predictions


# --- Modified EnKF_DA (LSTM-based) to return variance ---
# --- Modified EnKF_DA (LSTM-based) to return variance and predictions ---
def EnKF_DA(ground_truth, observations, r_values, model, device=None,
            ensemble_size=20, ensemble_size_total=100, seq_length=5,
            xlen=2.0, nx=100, dt=0.01, mu=0.01, partial_obs=0.6,
            nt_total=50, delta=1e-3, localization_radius=3, inflation=1.02,
            save_path="./results"):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.random.seed(10)
    I = np.eye(nx)
    ones = np.ones(ensemble_size)
    ones_total = np.ones(ensemble_size_total)

    all_analysis_errors = np.zeros((len(r_values), nt_total - 2))
    all_forecast_errors = np.zeros((len(r_values), nt_total - 2))
    all_analysis_variance = np.zeros((len(r_values), nt_total - 2))
    all_predictions = np.zeros((len(r_values), nx, nt_total - 2))

    L = create_localization_matrix(localization_radius, nx)

    for idx, r in enumerate(r_values):
        print(f"Processing r = {r} in EnKF-LSTM ({idx+1}/{len(r_values)})")

        analysis_errors = []
        forecast_errors = []
        analysis_variances = []
        predictions = []

        sigma_o = 0.01
        ny = int(round(partial_obs * nx))
        R_k = (sigma_o ** 2) * np.eye(ny)

        X1_b_sol = solve1dburgers(xlen, nx, 1, dt, mu, delta)
        xt_k = ground_truth[:, 1].copy()

        Xf_total = np.outer(ones_total, X1_b_sol[:, 1]) + 0.05 * np.random.randn(ensemble_size_total, nx)
        ind = np.random.permutation(ensemble_size_total)[:ensemble_size]
        Xf_k = Xf_total[ind, :].copy().T

        for k in range(nt_total - 2):
            current_step = k + 1
            xf_k = np.mean(Xf_k, axis=1)
            Pf_k = L * np.cov(Xf_k)

            forecast_error = np.linalg.norm(xf_k - xt_k)
            forecast_errors.append(forecast_error)

            obs_comp = np.random.permutation(ny)
            H_k = I[obs_comp, :]
            y_k = H_k @ observations[:, current_step]

            Eobs_k = sigma_o * np.random.randn(ny, ensemble_size)
            Yobs_k = np.outer(y_k, np.ones(ensemble_size)) + Eobs_k
            D_k = Yobs_k - H_k @ Xf_k
            IN_k = R_k + H_k @ Pf_k @ H_k.T
            Z_k = np.linalg.solve(IN_k, D_k)

            Xa_k = Xf_k + Pf_k @ H_k.T @ Z_k
            xa_k = np.mean(Xa_k, axis=1)
            DXa_k = Xa_k - np.outer(xa_k, ones)
            Xa_k = np.outer(xa_k, ones) + inflation * DXa_k

            analysis_error = np.linalg.norm(xa_k - xt_k)
            analysis_errors.append(analysis_error)

            analysis_variance = np.mean(np.var(Xa_k, axis=1))
            analysis_variances.append(analysis_variance)

            predictions.append(xa_k)

            X_a = xa_k.reshape(nx, 1)

            if k == 0:
                X1_b_sol[:, [1]] = X_a
            else:
                X1_b_sol = np.hstack((X1_b_sol, X_a))

            X_ensemble_forecast = np.zeros((nx, ensemble_size))

            for e in range(ensemble_size):
                current_state = Xa_k[:, e].reshape(nx, 1)

                if current_step >= seq_length:
                    temp_state = X1_b_sol.copy()
                    temp_state[:, -1] = current_state.flatten()
                    input_tensor = prepare_input(temp_state, current_step, seq_length, device)
                    with torch.no_grad():
                        u_pred = model(input_tensor)
                    X_ensemble_forecast[:, e] = u_pred.reshape(nx).cpu().numpy()
                else:
                    x_forecast = solve_one_step(current_state.flatten(), mu, xlen, nx, dt)
                    X_ensemble_forecast[:, e] = x_forecast

            Xf_k = X_ensemble_forecast
            xt_k = ground_truth[:, current_step + 1]

        all_analysis_errors[idx, :] = analysis_errors
        all_forecast_errors[idx, :] = forecast_errors
        all_analysis_variance[idx, :] = analysis_variances
        all_predictions[idx, :, :] = np.array(predictions).T

    return all_analysis_errors, all_forecast_errors, all_analysis_variance, all_predictions

def BLUE_DA_LSTM(ground_truth, observations, r_values, model, device=None,
            seq_length=5, xlen=2.0, nx=100, dt=0.01, mu=0.01, nt_total=50, delta=1e-3):
    
    all_analysis_errors = np.zeros((len(r_values), nt_total-2))
    
    for idx, r in enumerate(r_values):
        print(f"Processing r = {r} in BLUE ({idx+1}/{len(r_values)})")
        
        analysis_errors = []
        
        sigma_o = 0.01
        sigma_b = np.sqrt(r) * sigma_o
        
        W_mat = (r / (r + 1)) * np.eye(nx)
        
        # Initial background state
        X1_b_sol = solve1dburgers(xlen, nx, 1, dt, mu, delta)
        X_b = X1_b_sol[:, [1]]
        
        # Initial analysis
        X_a = X_b + W_mat @ (observations[:, [1]] - X_b)
        
        error = np.sqrt(np.mean((X_a - ground_truth[:, [1]])**2))
        analysis_errors.append(error)
        
        # Store analysis state as the first updated state
        X1_b_sol[:, [1]] = X_a
        
        for step in range(nt_total - 3):
            current_step = step + 2  # Start from timestep 2
            
            if current_step < 10:
                # Use observations to evolve using numerical scheme (or baseline physics)
                x_b_array = solve_one_step(X_a.flatten(), mu, xlen, nx, dt)
                X_b = x_b_array.reshape(nx, 1)
            elif current_step >= seq_length:
                # Use LSTM model
                input_tensor = prepare_input(X1_b_sol, current_step, seq_length, device)
                with torch.no_grad():
                    u_pred = model(input_tensor)
                X_b = u_pred.reshape(nx, 1).cpu().numpy()
            else:
                # Fallback: physics model if model input length is not ready
                x_b_array = solve_one_step(X_a.flatten(), mu, xlen, nx, dt)
                X_b = x_b_array.reshape(nx, 1)
            
            # Apply BLUE analysis step
            X_a = X_b + W_mat @ (observations[:, [current_step]] - X_b)
            
            # Compute analysis error
            error = np.sqrt(np.mean((X_a - ground_truth[:, [current_step]])**2))
            analysis_errors.append(error)
            
            # Append current analysis to the trajectory
            X1_b_sol = np.hstack((X1_b_sol, X_a))
        
        all_analysis_errors[idx, :] = analysis_errors
    
    return all_analysis_errors


import numpy as np

def BLUE_DA_RK4(ground_truth, observations, r_values, model=None, device=None,
                seq_length=5, xlen=2.0, nx=100, dt=0.01, mu=0.01, nt_total=50, delta=1e-3):
    
    all_analysis_errors = np.zeros((len(r_values), nt_total - 2))

    for idx, r in enumerate(r_values):
        print(f"Processing r = {r} in BLUE-RK4 ({idx+1}/{len(r_values)})")
        
        analysis_errors = []

        sigma_o = 0.01
        sigma_b = np.sqrt(r) * sigma_o
        
        # BLUE weight matrix
        W_mat = (r / (r + 1)) * np.eye(nx)
        
        # Initial background trajectory using RK4 solver
        X1_b_sol = solve1dburgers(xlen, nx, 1, dt, mu, delta)
        X_b = X1_b_sol[:, [1]]
        
        # First BLUE update
        X_a = X_b + W_mat @ (observations[:, [1]] - X_b)
        error = np.sqrt(np.mean((X_a - ground_truth[:, [1]])**2))
        analysis_errors.append(error)

        # Replace the initial state
        X1_b_sol[:, [1]] = X_a

        for k in range(nt_total - 3):
            current_step = k + 2

            # Forecast using RK4 one-step propagation
            x_b_array = solve_one_step(X_a.flatten(), mu, xlen, nx, dt)
            X_b = x_b_array.reshape(nx, 1)

            # BLUE analysis step
            X_a = X_b + W_mat @ (observations[:, [current_step]] - X_b)

            # Compute error
            error = np.sqrt(np.mean((X_a - ground_truth[:, [current_step]])**2))
            analysis_errors.append(error)

            # Update internal background state (optional if not used further)
            X1_b_sol = np.hstack((X1_b_sol, X_a))

        all_analysis_errors[idx, :] = analysis_errors

    return all_analysis_errors
