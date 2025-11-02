import numpy as np
import torch 
import matplotlib.pyplot as plt
from data_load import analytical1dburgers, add_noise_to_truth
from model_da import BLUE_DA, BLUE_DA_RK4, load_trained_model
from matplotlib.ticker import FormatStrFormatter
from scipy.io import savemat
import os

if __name__ == "__main__":
    
    # Simulation Parameters
    seq_length = 10
    delta = 1e-3
    xlen = 2 * np.pi
    nx = 1000
    nt_total = 5000
    dt = 0.01
    mu = 0.01
    dx = xlen / (nx - 1)
    c1 = 3
    
    # CFL check
    CFL_conv = 0.5
    CFL_diff = 0.1
    dt_stable = min(CFL_conv * dx / c1, CFL_diff * dx**2 / mu)
    if dt > dt_stable:
        print(f"Warning: dt={dt} too large. Adjusting to dt={dt_stable:.6e}")
        dt = dt_stable
    print(f"Using dt = {dt:.6e} (CFL-constrained)")

    # LSTM model
    hidden_size = 256
    num_layers = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model("lstm_burgers_99_1_split_6x256_lr5e-5_exp1.pth", input_size=nx, device=device,
                                hidden_size=hidden_size, num_layers=num_layers)
    
    # r values to test
    r_values = np.linspace(2, 500, 4, dtype=int)
    sigma_o = 0.01
    x = np.linspace(0, xlen, nx, endpoint=False)
    
    save_path = "./final_results"
    os.makedirs(save_path, exist_ok=True)

    # Ground truth + noisy observations
    ground_truth = analytical1dburgers(c1, xlen, nx, nt_total, dt, mu)
    observations = add_noise_to_truth(ground_truth, sigma_o)

    # Run BLUE with LSTM
    print("Running BLUE with LSTM...")
    blue_lstm_errors = BLUE_DA(
        ground_truth, observations, r_values, model, device,
        seq_length, xlen, nx, dt, mu, nt_total, delta
    )

    # Run BLUE with RK4
    print("Running BLUE with RK4...")
    blue_rk4_errors = BLUE_DA_RK4(
        ground_truth, observations, r_values, model=None, device=None,
        seq_length=seq_length, xlen=xlen, nx=nx, dt=dt, mu=mu, nt_total=nt_total, delta=delta
    )

    # Time axis
    time_steps = np.arange(blue_lstm_errors.shape[1]) * dt

    # --- Plot 1: Error Evolution Over Time ---
    plt.figure(figsize=(10, 6))
    for i, r in enumerate(r_values):
        plt.plot(time_steps, blue_lstm_errors[i], label=f'BLUE-LSTM, r={r}', linestyle='--')
        plt.plot(time_steps, blue_rk4_errors[i], label=f'BLUE-RK4, r={r}', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.title('Error Evolution Over Time (BLUE-RK4 vs BLUE-LSTM)')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(fontsize='small', ncol=2)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    plt.savefig(f'{save_path}/blue_lstm_vs_rk4_error_evolution.png', dpi=300)
    plt.close()

    # --- Plot 2: Average RMSE vs r ---
    blue_lstm_avg = np.mean(blue_lstm_errors, axis=1)
    blue_rk4_avg = np.mean(blue_rk4_errors, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(r_values, blue_lstm_avg, 'o--', label='BLUE-LSTM Avg RMSE')
    plt.plot(r_values, blue_rk4_avg, 's-', label='BLUE-RK4 Avg RMSE')
    plt.xlabel('r value (variance ratio)')
    plt.ylabel('Average RMSE')
    plt.title('Average RMSE vs r Value')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}/blue_lstm_vs_rk4_avg_rmse.png', dpi=300)
    plt.close()

    # Save results
    savemat(f'{save_path}/blue_lstm_vs_rk4.mat', {
        'r_values': r_values,
        'time_steps': time_steps,
        'blue_lstm_errors': blue_lstm_errors,
        'blue_rk4_errors': blue_rk4_errors,
        'blue_lstm_avg': blue_lstm_avg,
        'blue_rk4_avg': blue_rk4_avg
    })

    print("Plots and data saved successfully.")
