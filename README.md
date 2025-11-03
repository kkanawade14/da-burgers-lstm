
# da-burgers-lstm
Physics-informed LSTM model for 1D Burgers’ equation data assimilation
=======
# Burgers' Equation Data Assimilation Framework

This repository contains implementations of data assimilation techniques for the 1D Burgers' equation using both traditional numerical methods and deep learning approaches. The framework offers two data assimilation methods: BLUE (Best Linear Unbiased Estimator) and Kalman Filter, both coupled with an LSTM neural network model for state forecasting.

## Problem Setup

We are solving the 1D Burgers' equation, a fundamental nonlinear PDE that serves as a simplified model for fluid dynamics:

```
u_t + u*u_x - ν*u_xx = 0
```

Where:
- `u` is the velocity field
- `ν` is the viscosity coefficient
- Subscripts denote partial derivatives

### Numerical Details

The system is set up with the following parameters:

- **Domain**: x ∈ [0, 2π]
- **Spatial discretization**: nx = 1000 grid points
- **Temporal range**: nt_total = 5000 time steps
- **Time step**: dt = 3.955749e-04 (CFL-constrained)
- **Viscosity**: μ = 0.01
- **Spatial step size**: dx = 2π/999

The time step is constrained by the CFL (Courant-Friedrichs-Lewy) condition to ensure numerical stability:
- Convective limit: dt_conv = 0.5 * dx / c1 (where c1 = 3 is the estimated wave speed)
- Diffusive limit: dt_diff = 0.1 * dx² / μ
- Final time step: dt = min(dt_conv, dt_diff) = 3.955749e-04

## Data Assimilation Methods

### 1. BLUE Method

The BLUE method uses a static weight matrix to blend model forecasts with observations. The weight matrix is derived from the ratio of background to observation error covariances:

```
W = (r/(r+1)) * I
```

Where:
- `r` is the ratio of background to observation error variance
- `I` is the identity matrix

The analysis update is given by:
```
x_a = x_b + W * (y - x_b)
```

### 2. Kalman Filter Method

The Kalman Filter extends the BLUE method by including dynamic error covariance evolution. Key equations:

- **Forecast step**:
  ```
  x_f = M(x_a)
  P_f = M*P_a*M^T + Q
  ```

- **Analysis step**:
  ```
  K = P_f * (P_f + R)^-1
  x_a = x_f + K * (y - x_f)
  P_a = (I - K) * P_f
  ```

Where:
- `M` is the forecast model (LSTM or numerical solver)
- `P_f` and `P_a` are the forecast and analysis error covariances
- `Q` is the model error covariance
- `R` is the observation error covariance
- `K` is the Kalman gain matrix

## LSTM Neural Network Model

A recurrent neural network is used to provide fast forecasts of the system state:

- **Architecture**: LSTM (Long Short-Term Memory)
- **Hidden size**: 256 units
- **Number of layers**: 6
- **Sequence length**: 10 time steps

The LSTM is trained to predict the next state given a sequence of previous states. During data assimilation, the LSTM model is used after accumulating enough history (typically 5 time steps), with the numerical solver being used for the initial steps.

## Experiment Setup

The experiments test various values of the error variance ratio `r` to determine optimal assimilation weights:

```
r_values = [2, 12, 22, 32, 42, 52, 62, 72, 100, 200, 300, 400]
```

For each value of `r`:
1. An initial background state is generated using the numerical solver
2. Data assimilation is performed using both BLUE and Kalman Filter methods
3. Errors are calculated against a known ground truth
4. Performance is evaluated based on RMSE (Root Mean Square Error)

## Implementation Details

### State Propagation

- **Numerical solver**: For the first few time steps and when high accuracy is required
- **LSTM model**: For faster forecasting once sufficient state history is accumulated

The LSTM input is prepared using:
```python
input_tensor = prepare_input(X1_b_sol, step+3-5, seq_length, device)
```

This extracts a sequence of `seq_length` states from the solution history `X1_b_sol`, starting from index `step+3-5`.

### Error Calculation

RMSE is calculated between the analysis state and the ground truth:
```python
error = np.sqrt(np.mean((X_a - ground_truth[:, [step+2]])**2))
```

## Usage

To run the comparison between BLUE and Kalman Filter:

```python
blue_errors, kalman_errors, fig = run_comparison(
    ground_truth, observations, r_values, model, device,
    seq_length, xlen, nx, dt, mu, nt_total, delta
)
```

## Results

The results compare the performance of BLUE and Kalman Filter methods across different values of `r`. Key findings include:

- How error evolves over time for both methods
- Effect of the error variance ratio on assimilation performance
- Conditions under which the Kalman Filter outperforms the BLUE method

