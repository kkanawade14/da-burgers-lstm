import numpy as np
import torch
import matplotlib.pyplot as plt
from data_load import analytical1dburgers, add_noise_to_truth
from model_da import EnKF_DA, EnKF_DA_RK4, load_trained_model
from matplotlib.ticker import FormatStrFormatter
from scipy.io import savemat
import os

if __name__ == "__main__":

    seq_length = 10
    delta = 1e-3
    xlen = 2 * np.pi
    nx = 1000
    nt_total = 5000
    dt = 0.01
    mu = 0.01
    dx = xlen / (nx - 1)
    c1 = 3

    CFL_conv = 0.5
    CFL_diff = 0.1
    dt_stable = min(CFL_conv * dx / c1, CFL_diff * dx**2 / mu)
    if dt > dt_stable:
        print(f"Warning: dt={dt} too large. Adjusting to dt={dt_stable:.6e}")
        dt = dt_stable
    print(f"Using dt = {dt:.6e} (CFL-constrained)")

    hidden_size = 256
    num_layers = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model("lstm_burgers_99_1_split_6x256_lr5e-5.pth", input_size=nx, device=device,
                                hidden_size=hidden_size, num_layers=num_layers)

    r_values = np.array([2])
    sigma_o = 0.01
    x = np.linspace(0, xlen, nx, endpoint=False)

    save_path = "./final_results_enkf"
    os.makedirs(save_path, exist_ok=True)

    ground_truth = analytical1dburgers(c1, xlen, nx, nt_total, dt, mu)
    observations = add_noise_to_truth(ground_truth, sigma_o)

    print("Running EnKF with LSTM...")
    enkf_lstm_errors, enkf_lstm_forecast, enkf_lstm_var, enkf_lstm_pred = EnKF_DA(
        ground_truth, observations, r_values, model, device,
        ensemble_size=15, ensemble_size_total=100, seq_length=seq_length,
        xlen=xlen, nx=nx, dt=dt, mu=mu, nt_total=nt_total, delta=delta,
        localization_radius=3, inflation=1.02
    )

    print("Running EnKF with RK4...")
    enkf_rk4_errors, enkf_rk4_forecast, enkf_rk4_var, enkf_rk4_pred = EnKF_DA_RK4(
        ground_truth, observations, r_values, model, device,
        ensemble_size=15, ensemble_size_total=100, seq_length=seq_length,
        xlen=xlen, nx=nx, dt=dt, mu=mu, nt_total=nt_total, delta=delta,
        localization_radius=3, inflation=1.02
    )

    time_steps = np.arange(enkf_lstm_errors.shape[1]) * dt

    # State comparison at selected time indices (only for first r value)
    plot_indices = [500, 2000, 4000]
    for t_idx in plot_indices:
        idx = t_idx - 2
        plt.figure(figsize=(10, 6))
        plt.plot(x, ground_truth[:, t_idx], label="Ground Truth", linewidth=2)
        plt.plot(x, enkf_rk4_pred[0, :, idx], label="EnKF-RK4", linestyle="--")
        plt.plot(x, enkf_lstm_pred[0, :, idx], label="EnKF-LSTM", linestyle=":")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title(f"State Comparison at t = {t_idx * dt:.2f}, r = {r_values[0]}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}/state_comparison_r{r_values[0]}_t{t_idx}.png", dpi=300)
        plt.close()

    # Variance comparison across all r values
    plt.figure(figsize=(10, 6))
    for i, r in enumerate(r_values):
        plt.plot(time_steps, enkf_lstm_var[i], linestyle="--", label=f"LSTM, r={r}")
        plt.plot(time_steps, enkf_rk4_var[i], linestyle="-", label=f"RK4, r={r}")
    plt.xlabel("Time")
    plt.ylabel("Ensemble Variance")
    plt.title("Analysis Variance Evolution for Different r Values")
    plt.yscale("log")
    plt.grid(True)
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(f"{save_path}/variance_comparison_all_r.png", dpi=300)
    plt.close()

    # Plot: Error Evolution Over Time (All r values)
    plt.figure(figsize=(10, 6))
    for i, r in enumerate(r_values):
        plt.plot(time_steps, enkf_lstm_errors[i], label=f'EnKF-LSTM, r={r}', linestyle='--')
        plt.plot(time_steps, enkf_rk4_errors[i], label=f'EnKF-RK4, r={r}', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.title('Error Evolution Over Time (EnKF-RK4 vs EnKF-LSTM)')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(fontsize='small', ncol=2)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.tight_layout()
    plt.savefig(f'{save_path}/enkf_lstm_vs_rk4_error_evolution.png', dpi=300)
    plt.close()

    # Plot: Average RMSE vs r
    enkf_lstm_avg = np.mean(enkf_lstm_errors, axis=1)
    enkf_rk4_avg = np.mean(enkf_rk4_errors, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(r_values, enkf_lstm_avg, 'o--', label='EnKF-LSTM Avg RMSE')
    plt.plot(r_values, enkf_rk4_avg, 's-', label='EnKF-RK4 Avg RMSE')
    plt.xlabel('r value (variance ratio)')
    plt.ylabel('Average RMSE')
    plt.title('Average RMSE vs r Value')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}/enkf_lstm_vs_rk4_avg_rmse.png', dpi=300)
    plt.close()

    # Save results
    savemat(f'{save_path}/enkf_lstm_vs_rk4.mat', {
        'r_values': r_values,
        'time_steps': time_steps,
        'enkf_lstm_errors': enkf_lstm_errors,
        'enkf_rk4_errors': enkf_rk4_errors,
        'enkf_lstm_avg': enkf_lstm_avg,
        'enkf_rk4_avg': enkf_rk4_avg,
        'enkf_lstm_var': enkf_lstm_var,
        'enkf_rk4_var': enkf_rk4_var
    })

    print("All EnKF plots and data saved successfully.")