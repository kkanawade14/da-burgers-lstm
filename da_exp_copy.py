import numpy as np
import torch 
import matplotlib.pyplot as plt
from data_load import analytical1dburgers
from data_load import add_noise_to_truth, burgers_jacobian
from scipy.io import savemat
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import FuncAnimation
from model_da import BLUE_DA, load_trained_model, EnKF_DA_RK4, plot_analysis_variance, visualize_state_evolution
import os


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

    hidden_size = 256
    num_layers = 6

    c1 = 3 
    # CFL Conditions
    CFL_conv = 0.5  # convective term
    CFL_diff = 0.1  # diffusive term

    # Compute Maximum Allowed Time Step
    dt_conv = CFL_conv * dx / c1
    dt_diff = CFL_diff * dx**2 / mu

    # most restrictive time step
    dt_stable = min(dt_conv, dt_diff)

    if dt > dt_stable:
        print(f"Warning: dt={dt} is too large! Adjusting to dt={dt_stable:.6e} for stability.")
        dt = dt_stable

    print(f"Using dt = {dt:.6e} (CFL-constrained)")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model("lstm_burgers_model.pth", input_size=nx, device=device, 
                             hidden_size=hidden_size, num_layers=num_layers)

    
    # r_values = np.concatenate([np.arange(2, 90, 40), np.arange(100, 500, 250)])
    # r_values = np.concatenate([np.arange(2, 82, 40) ])
    r_values = np.concatenate([np.arange(2, 43, 40) ])
    
    sigma_o = 0.01
    
    x = np.linspace(0, xlen, nx, endpoint=False)

    
    save_path = "./results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

   
    ground_truth = analytical1dburgers(c1, xlen, nx, nt_total, dt, mu)
    observations = add_noise_to_truth(ground_truth, sigma_o)


    print("Running BLUE data assimilation...")
    blue_errors = BLUE_DA(
        ground_truth, observations, r_values, model, device,
        seq_length, xlen, nx, dt, mu, nt_total, delta
    )


    print("Running EnKF data assimilation...")
    enkf_analysis_errors, enkf_forecast_errors, enkf_analysis_variance, \
    selected_analysis_states = EnKF_DA_RK4(
        ground_truth, observations, r_values, model, device,
        ensemble_size=500, ensemble_size_total=1000, seq_length=seq_length, 
        xlen=xlen, nx=nx, dt=dt, mu=mu, nt_total=nt_total, delta=delta,
        localization_radius=(2 * np.pi)/20, inflation=1.02,
        save_path=save_path
    )
    

    time_steps = np.arange(blue_errors.shape[1]) * dt

    # Calculate average RMSE for each method
    blue_avg = np.mean(blue_errors, axis=1)
    enkf_analysis_avg = np.mean(enkf_analysis_errors, axis=1)
    enkf_forecast_avg = np.mean(enkf_forecast_errors, axis=1)
    
    # --- Plot 6: State evolution for selected r values ---
   
    print(f"Creating state evolution visualizations for r values")
    
    for i, r_idx in enumerate(r_values):
        visualize_state_evolution(ground_truth, observations, 
                                selected_analysis_states[i], r_values[i], 
                                nt_total, nx, save_path)
    
    # Save data
    data_to_save = {
        'r_values': r_values,
        'time_steps': time_steps,
        'blue_errors': blue_errors,
        'enkf_analysis_errors': enkf_analysis_errors,
        'enkf_forecast_errors': enkf_forecast_errors,
        'enkf_analysis_variance': enkf_analysis_variance,
        'blue_avg': blue_avg,
        'enkf_analysis_avg': enkf_analysis_avg,
        'enkf_forecast_avg': enkf_forecast_avg
    }
    savemat(f'{save_path}/data_assimilation_comparison.mat', data_to_save)
    print(f"Data saved to {save_path}/data_assimilation_comparison.mat")

    print("Creating plots...")
    
    # --- Plot 1: All BLUE errors ---
    markers = ['o', 's', '^', 'v', 'p', '*', 'h', 'H', 'D', 'd', '|', '_', 'x', '+', '.']
    num_styles = len(markers)
    cmap = plt.get_cmap('tab20b', r_values.shape[0])
    
    plt.figure(figsize=(10, 6))
    for i in range(len(r_values)):
        marker = markers[i % num_styles]
        color = cmap(i)
        plt.plot(time_steps, blue_errors[i], linestyle='--', marker=marker, markevery=max(1, len(time_steps)//20),
                color=color, label=f'BLUE, r={r_values[i]:.1e}')
    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.title('BLUE Error Evolution Over Time (All r values)')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(fontsize='small', ncol=2)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    plt.savefig(f'{save_path}/blue_error_all_r.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 2: All EnKF Analysis errors ---
    plt.figure(figsize=(10, 6))
    for i in range(len(r_values)):
        marker = markers[i % num_styles]
        color = cmap(i)
        plt.plot(time_steps, enkf_analysis_errors[i], linestyle='-', marker=marker, markevery=max(1, len(time_steps)//20),
                color=color, label=f'EnKF, r={r_values[i]:.1e}')
    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.title('EnKF Analysis Error Evolution Over Time (All r values)')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(fontsize='small', ncol=2)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    plt.savefig(f'{save_path}/enkf_analysis_error_all_r.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 3: All EnKF Forecast errors ---
    plt.figure(figsize=(10, 6))
    for i in range(len(r_values)):
        marker = markers[i % num_styles]
        color = cmap(i)
        plt.plot(time_steps, enkf_forecast_errors[i], linestyle=':', marker=marker, markevery=max(1, len(time_steps)//20),
                color=color, label=f'EnKF Fcst, r={r_values[i]:.1e}')
    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.title('EnKF Forecast Error Evolution Over Time (All r values)')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(fontsize='small', ncol=2)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    plt.savefig(f'{save_path}/enkf_forecast_error_all_r.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 4: Average RMSE vs r ---
    plt.figure(figsize=(8, 6))
    plt.plot(r_values, blue_avg, 'o-', label='BLUE Avg RMSE')
    plt.plot(r_values, enkf_analysis_avg, 's-', label='EnKF Analysis Avg RMSE')
    plt.plot(r_values, enkf_forecast_avg, '^--', label='EnKF Forecast Avg RMSE')
    plt.xlabel('r value (variance ratio)')
    plt.ylabel('Average RMSE')
    plt.title('Average Error vs r Value')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}/avg_rmse_vs_r.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 5: Analysis variance evolution ---
    plot_analysis_variance(enkf_analysis_variance, r_values, dt, save_path)

    

    
    print("All plots saved successfully.")