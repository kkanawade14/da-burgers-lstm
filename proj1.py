# run_lstm_teacher_forced_spacetime.py

import numpy as np
import torch
import time
import matplotlib
matplotlib.use("Agg")  # non‑interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import LSTMBurgers
from data_load import analytical1dburgers

# ---------------- Parameters ----------------
xlen        = 2 * np.pi
nx          = 1000
nt_total    = 5000
mu          = 0.01
dt = 0.01
c1          = 3
dx = xlen / (nx - 1)
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
seq_length  = 10

model_path  = "lstm_burgers_99_1_split_6x256_lr6e-4.pth"
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Helpers ----------------
def load_trained_model(model_path, input_size, hidden_size=256, num_layers=6, device="cpu"):
    model = LSTMBurgers(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()

def prepare_input(data, idx, seq_length, device="cpu"):
    window = data[:, idx:idx+seq_length].T          # (seq_length, nx)
    return (torch.tensor(window, dtype=torch.float32)
                .unsqueeze(0)
                .to(device))                        # (1, seq_length, nx)

def predict_teacher_forced(model, data, seq_length, total_steps, device):
    preds = []
    for t in range(seq_length, total_steps):
        inp = prepare_input(data, t, seq_length, device)
        with torch.no_grad():
            out = model(inp).squeeze(0).cpu().numpy()  # (nx,)
        preds.append(out)
        if (t - seq_length + 1) % 500 == 0:
            print(f"  ↳ Predicted step {t}/{total_steps-1}")
    return np.vstack(preds)  # (nt-seq_length, nx)

def compute_rmse(pred, true):
    return np.sqrt(np.mean((pred - true)**2))


# ---------------- Main ----------------
if __name__ == "__main__":
    t0 = time.time()

    print("1) Generating analytical Burgers solution …")
    data = analytical1dburgers(c1, xlen, nx, nt_total, dt, mu)  # (nx, nt)

    print("2) Loading trained LSTM model …")
    model = load_trained_model(model_path, input_size=nx, device=device)

    print("3) Running teacher‑forced prediction …")
    predicted = predict_teacher_forced(model, data, seq_length, nt_total, device)
    observations = data[:, seq_length:].T  # (nt-seq_length, nx)

    # Save raw arrays
    np.save("observations_tf.npy", observations)
    np.save("predictions_tf.npy", predicted)
    print("   • observations_tf.npy, predictions_tf.npy saved")

    # Compute and print RMSE
    rmse = compute_rmse(predicted, observations)
    print(f"4) Teacher‑forced RMSE: {rmse:.6e}")

    # ---------------- Spacetime Heatmaps ----------------
    T_total = dt * (nt_total - seq_length)
    x = np.linspace(0, xlen, nx)
    t = np.linspace(dt*seq_length, dt*nt_total, observations.shape[0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    mats   = [observations, predicted, np.abs(observations - predicted)]
    titles = ["Observations", "Predictions", "Absolute Error"]
    cmaps  = ["viridis", "viridis", "hot"]

    for ax, mat, title, cmap in zip(axes, mats, titles, cmaps):
        im = ax.imshow(
            mat, 
            origin="lower", 
            aspect="auto",
            extent=[0, xlen, t[0], t[-1]],
            cmap=cmap
        )
        ax.set_xlabel("Spatial coordinate $x$")
        ax.set_ylabel("Time $t$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Velocity $u$")
    plt.savefig("spacetime_heatmaps2.png", dpi=300)
    print("   • spacetime_heatmaps.png saved")

    # ---------------- Animation ----------------
    # nt_anim, nx_anim = observations.shape
    # fig, ax = plt.subplots(figsize=(10, 5))
    # line_obs, = ax.plot([], [], lw=2, label="True")
    # line_pred, = ax.plot([], [], lw=2, label="Pred")
    # ax.set_xlim(0, nx_anim)
    # ax.set_ylim(observations.min(), observations.max())
    # ax.set_xlabel("Spatial index")
    # ax.set_ylabel("Velocity $u$")
    # ax.legend()

    # def init():
    #     line_obs.set_data([], [])
    #     line_pred.set_data([], [])
    #     return line_obs, line_pred

    # def animate(i):
    #     x = np.arange(nx_anim)
    #     line_obs.set_data(x, observations[i])
    #     line_pred.set_data(x, predicted[i])
    #     ax.set_title(f"Time step {i+seq_length}")
    #     return line_obs, line_pred

    # ani = animation.FuncAnimation(
    #     fig, animate, frames=nt_anim,
    #     init_func=init, blit=True, interval=50
    # )
    # ani.save("lstm_tf_spacetime.gif", writer="pillow")
    # print("   • lstm_tf_spacetime.gif saved")

    print(f"✅ Done in {time.time() - t0:.1f}s")
