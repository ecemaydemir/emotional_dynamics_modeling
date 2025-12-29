import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ==========================================
# 1. MODEL I: LINEAR DECAY DYNAMICS
# ==========================================
# Equation: dy/dt = -ky
# Description: Models natural emotional resilience/fading over time.

def linear_decay(y, t, k):
    return -k * y

# Parameters
k = 0.5   # Decay rate (Resilience)
y0 = 10.0 # Initial emotional intensity
t = np.linspace(0, 20, 100)

# Solve ODE
y_linear = odeint(linear_decay, y0, t, args=(k,))

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(t, y_linear, label=f'Decay Rate k={k}', color='navy', linewidth=2)
plt.fill_between(t, y_linear.flatten(), color='navy', alpha=0.1)
plt.title('Model I: Natural Emotional Decay')
plt.xlabel('Time')
plt.ylabel('Emotional Intensity')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# ==========================================
# 2. MODEL II: ENVIRONMENTAL FORCING
# ==========================================
# Equation: dy/dt = -ky + c
# Description: Models response to constant environmental stimuli.

def forced_response(y, t, k, c):
    return -k * y + c

# Parameters
c = 2.0   # Constant external input (Stimuli)
y0_forced = 0.0 # Start from neutral state
equilibrium = c / k

# Solve ODE
y_forced = odeint(forced_response, y0_forced, t, args=(k, c))

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(t, y_forced, label='Intensity Response', color='firebrick', linewidth=2)
plt.axhline(y=equilibrium, color='teal', linestyle='--', label=f'Equilibrium (y={equilibrium})')
plt.title('Model II: Response to Constant Stimuli')
plt.xlabel('Time')
plt.ylabel('Emotional Intensity')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# ==========================================
# 3. MODEL III: NONLINEAR REGULATION
# ==========================================
# Equation: dy/dt = -ky + ay(1-y)
# Description: Logistic model representing internal regulation limits.

def nonlinear_regulation(y, t, k, a):
    return -k * y + a * y * (1 - y)

# Parameters
k_nonlin = 0.2
a = 0.8
y0_values = [0.1, 0.5, 1.2] # Different initial states
t_long = np.linspace(0, 30, 100)
stable_eq = 1 - (k_nonlin/a)

# Visualization
plt.figure(figsize=(10, 6))
for y0 in y0_values:
    y_sol = odeint(nonlinear_regulation, y0, t_long, args=(k_nonlin, a))
    plt.plot(t_long, y_sol, label=f'Initial y₀={y0}')

plt.axhline(y=stable_eq, color='black', linestyle='--', alpha=0.7, label=f'Stable Eq (y≈{stable_eq:.2f})')
plt.title('Model III: Nonlinear Stability Analysis')
plt.xlabel('Time')
plt.ylabel('Emotional Intensity')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# ==========================================
# 4. MODEL IV: COUPLED EMOTIONAL INTERACTION
# ==========================================
# Equations:
# dx/dt = -ax + by (x is driven by y)
# dy/dt = -cy      (y decays naturally)

def coupled_system(z, t, a, b, c):
    x, y = z
    dxdt = -a * x + b * y
    dydt = -c * y
    return [dxdt, dydt]

# Parameters
params = (0.5, 0.5, 0.2) # a, b, c
initial_state = [5.0, 5.0] # [x0, y0]

# Solve ODE
solution = odeint(coupled_system, initial_state, t_long, args=params)
x_sol = solution[:, 0]
y_sol = solution[:, 1]

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(t_long, x_sol, label='Emotion X (Driven) - e.g. Fatigue', color='purple', linewidth=2)
plt.plot(t_long, y_sol, label='Emotion Y (Source) - e.g. Anxiety', color='orange', linestyle='--', linewidth=2)
plt.title('Model IV: Coupled Emotional Dynamics (Lag Effect)')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
