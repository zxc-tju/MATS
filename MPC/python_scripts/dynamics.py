import autograd.numpy as np
from autograd import jacobian


def linearize_dynamics(dynamics, x0, u0, dt):
    jac_x = jacobian(lambda x: dynamics(x, u0, dt), 0)
    jac_u = jacobian(lambda u: dynamics(x0, u, dt), 0)
    return jac_x(x0), jac_u(u0)


def discrete_dynamics(x, u, dt=0.02):
    ω, a = u[0], u[1]
    if np.abs(ω) > 1E-3:
        return np.array([
            x[0] + x[3]/ω * (np.sin(x[2] + ω*dt) - np.sin(x[2])) + a/ω * dt * np.sin(x[2] + ω*dt) + a/ω**2 * (np.cos(x[2] + ω*dt) - np.cos(x[2])),
            x[1] - x[3]/ω * (np.cos(x[2] + ω*dt) - np.cos(x[2])) - a/ω * dt * np.cos(x[2] + ω*dt) + a/ω**2 * (np.sin(x[2] + ω*dt) - np.sin(x[2])),
            x[2] + ω * dt,
            x[3] + a * dt
        ])
    else:
        return np.array([
            x[0] + x[3] * dt * np.cos(x[2]) + 0.5 * a * dt**2 * np.cos(x[2]),
            x[1] + x[3] * dt * np.sin(x[2]) + 0.5 * a * dt**2 * np.sin(x[2]),
            x[2],
            x[3] + a * dt
        ])
