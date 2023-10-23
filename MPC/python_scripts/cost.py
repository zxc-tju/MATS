import numpy as np
from autograd import grad
from scipy.interpolate import CubicSpline



from MPC.python_scripts.path_handling import find_spline_interval


def approx_contouring_error(P, path, spline_idx):
    X, Y, s = P
    x_ref = path['x'](s)
    y_ref = path['y'](s)
    Φ = path['heading'](s)
    return np.sin(Φ) * (X - x_ref) - np.cos(Φ) * (Y - y_ref)

def approx_lag_error(P, path, spline_idx):
    X, Y, s = P
    x_ref = path['x'](s)
    y_ref = path['y'](s)
    Φ = path['heading'](s)
    return -np.cos(Φ) * (X - x_ref) - np.sin(Φ) * (Y - y_ref)

def contouring_error_gradient(P, path, spline_idx):
    return grad(lambda P: approx_contouring_error(P, path, spline_idx))(P)

def lag_error_gradient(P, path, spline_idx):
    return grad(lambda P: approx_lag_error(P, path, spline_idx))(P)

def tracking_linear_term(P0, qc, ql, path, spline_idx):
    X0, Y0, s0 = P0
    ϵc0 = approx_contouring_error(P0, path, spline_idx)
    ϵl0 = approx_lag_error(P0, path, spline_idx)
    dϵc_dx, dϵc_dy, dϵc_ds = contouring_error_gradient(P0, path, spline_idx)
    dϵl_dx, dϵl_dy, dϵl_ds = lag_error_gradient(P0, path, spline_idx)

    cc = np.array([2 * qc * ϵc0 * dϵc_dx - 2 * qc * X0 * dϵc_dx**2 - 2 * qc * Y0 * dϵc_dx * dϵc_dy - 2 * qc * s0 * dϵc_dx * dϵc_ds,
                   2 * qc * ϵc0 * dϵc_dy - 2 * qc * Y0 * dϵc_dy**2 - 2 * qc * X0 * dϵc_dx * dϵc_dy - 2 * qc * s0 * dϵc_dy * dϵc_ds,
                   2 * qc * ϵc0 * dϵc_ds - 2 * qc * s0 * dϵc_ds**2 - 2 * qc * X0 * dϵc_dx * dϵc_ds - 2 * qc * Y0 * dϵc_dy * dϵc_ds])

    cl = np.array([2 * ql * ϵl0 * dϵl_dx - 2 * ql * X0 * dϵl_dx**2 - 2 * ql * Y0 * dϵl_dx * dϵl_dy - 2 * ql * s0 * dϵl_dx * dϵl_ds,
                   2 * ql * ϵl0 * dϵl_dy - 2 * ql * Y0 * dϵl_dy**2 - 2 * ql * X0 * dϵl_dx * dϵl_dy - 2 * ql * s0 * dϵl_dy * dϵl_ds,
                   2 * ql * ϵl0 * dϵl_ds - 2 * ql * s0 * dϵl_ds**2 - 2 * ql * X0 * dϵl_dx * dϵl_ds - 2 * ql * Y0 * dϵl_dy * dϵl_ds])

    return cc + cl

def tracking_quadratic_term(P0, qc, ql, path, spline_idx):
    dϵc_dx, dϵc_dy, dϵc_ds = contouring_error_gradient(P0, path, spline_idx)
    dϵl_dx, dϵl_dy, dϵl_ds = lag_error_gradient(P0, path, spline_idx)

    Qc = np.array([[qc * dϵc_dx**2, qc * dϵc_dx * dϵc_dy, qc * dϵc_dx * dϵc_ds],
                   [qc * dϵc_dx * dϵc_dy, qc * dϵc_dy**2, qc * dϵc_dy * dϵc_ds],
                   [qc * dϵc_dx * dϵc_ds, qc * dϵc_dy * dϵc_ds, qc * dϵc_ds**2]])

    Ql = np.array([[ql * dϵl_dx**2, ql * dϵl_dx * dϵl_dy, ql * dϵl_dx * dϵl_ds],
                   [ql * dϵl_dx * dϵl_dy, ql * dϵl_dy**2, ql * dϵl_dy * dϵl_ds],
                   [ql * dϵl_dx * dϵl_ds, ql * dϵl_dy * dϵl_ds, ql * dϵl_ds**2]])

    return Qc + Ql
#
# def tracking_cost_matrices(qs, vals):
#     S_q = vals['k_c'] + vals['n_modes'] * (vals['N'] - vals['k_c'])
#     cs = []
#     Γs = []
#     for k in range(S_q):
#         P0 = qs[:3, k]  # X, Y, s of the initial guess
#         spline_idx = find_spline_interval(P0[2], vals['path'])
#         if k in end_horizon_idces(vals):
#             ck = tracking_linear_term(P0, vals['qcterm'], vals['ql'], vals['path'], spline_idx)
#             Γk = tracking_quadratic_term(P0, vals['qcterm'], vals['ql'], vals['path'], spline_idx)
#         else:
#             ck = tracking_linear_term(P0, vals['qc'], vals['ql'], vals['path'], spline_idx)
#             Γk = tracking_quadratic_term(P0, vals['qc'], vals['ql'], vals['path'], spline_idx)
#         cs.append(ck)
#         Γs.append(Γk)
#     vals['c'] = np.concatenate(cs)
#     vals['Γ'] = block_diag(*Γs)
