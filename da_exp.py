import numpy as np
import torch 
import matplotlib.pyplot as plt
from data_load import analytical1dburgers
from data_load import add_noise_to_truth, burgers_jacobian
from scipy.io import savemat
from matplotlib.ticker import FormatStrFormatter
from model_da import EKF_DA, BLUE_DA, load_trained_model



if __name__ == "__main__":
    

    # parameters
    seq_length = 10
    delta = 1e-3
    
    xlen = 2 * np.pi  
    nx = 1000  
    nt_total = 5000
    dt = 0.01  
    mu = 0.01  
    dx = xlen / (nx - 1)

    hidden_size=256
    num_layers=6

    c1 = 3 
    # CFL Conditions
    CFL_conv = 0.5  #  convective term
    CFL_diff = 0.1  #  diffusive term

    # Compute Maximum Allowed Time Step
    dt_conv = CFL_conv * dx / c1
    dt_diff = CFL_diff * dx**2 / mu

    # most restrictive time step
    dt_stable = min(dt_conv, dt_diff)

    if dt > dt_stable:
        print(f"Warning: dt={dt} is too large! Adjusting to dt={dt_stable:.6e} for stability.")
        dt = dt_stable

    print(f"Using dt = {dt:.6e} (CFL-constrained)")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model("lstm_burgers_model.pth", input_size=nx,device=device, hidden_size=hidden_size, num_layers=num_layers)

    
    r_values = np.concatenate([np.arange(2, 90, 10), np.arange(100, 500, 100)])
    
    sigma_o = 0.01
    # sigma_b = np.sqrt(r) * sigma_o

    x = np.linspace(0,xlen,nx, endpoint=False)


    ground_truth = analytical1dburgers(c1,xlen, nx, nt_total, dt, mu)
    observations = add_noise_to_truth(ground_truth, sigma_o)

    # Extended Kalman Filter
    kalman_errors = EKF_DA(
        ground_truth, observations, r_values, model, device,
        seq_length, xlen, nx, dt, mu, nt_total, delta
    )

    #  BLUE 
    blue_errors = BLUE_DA(
        ground_truth, observations, r_values, model, device,
        seq_length, xlen, nx, dt, mu, nt_total, delta
    )
    
    
    time_steps = np.arange(blue_errors.shape[1]) * dt

    blue_avg = np.mean(blue_errors, axis=1)
    kalman_avg = np.mean(kalman_errors, axis=1)

    # Save data before plotting
    data_to_save = {
        'r_values': r_values,
        'time_steps': time_steps,
        'blue_errors': blue_errors,
        'kalman_errors': kalman_errors,
        'blue_avg': blue_avg,
        'kalman_avg': kalman_avg
    }
    savemat('error_analysis_data.mat', data_to_save)
    print("Data saved to error_analysis_data.mat")
    print("Plotting starts")


    plt.figure(figsize=(10, 8))
    markers = ['o', 's', '^', 'v', 'p', '*', 'h', 'H', 'D', 'd', '|', '_', 'x', '+', '.']
    num_styles = len(markers)
    cmap = plt.cm.get_cmap('tab20b', r_values.shape[0])

    # --- Plot 1: All BLUE errors ---
    plt.figure(figsize=(10, 6))
    for i in range(len(r_values)):
        marker = markers[i % num_styles]
        color = cmap(i)
        plt.plot(time_steps, blue_errors[i], linestyle='--', marker=marker, markevery=10,
                color=color, label=f'BLUE, r={r_values[i]:.1e}')
    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.title('BLUE Error Evolution Over Time (All r values)')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(fontsize='small', ncol=2)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    plt.savefig('blue_error_all_r.png', dpi=300, bbox_inches='tight')

    # --- Plot 2: All KF errors ---
    plt.figure(figsize=(10, 6))
    for i in range(len(r_values)):
        marker = markers[i % num_styles]
        color = cmap(i)
        plt.plot(time_steps, kalman_errors[i], linestyle='-', marker=marker, markevery=10,
                color=color, label=f'KF, r={r_values[i]:.1e}')
    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.title('Extended Kalman Filter Error Evolution Over Time (All r values)')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(fontsize='small', ncol=2)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    plt.savefig('kf_error_all_r.png', dpi=300, bbox_inches='tight')

    # --- Plot 3: Average RMSE vs r ---
    plt.figure(figsize=(8, 6))
    plt.plot(r_values, blue_avg, 'o-', label='BLUE Avg RMSE')
    plt.plot(r_values, kalman_avg, 's-', label='EKF Avg RMSE')
    plt.xlabel('r value (variance ratio)')
    plt.ylabel('Average RMSE')
    plt.title('Average Error vs r Value')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('avg_rmse_vs_r.png', dpi=300, bbox_inches='tight')

    print("All plots saved successfully.")
