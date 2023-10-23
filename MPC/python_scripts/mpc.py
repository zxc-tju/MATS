import numpy as np
import casadi as ca
from scipy.linalg import block_diag

from MPC.python_scripts.path_handling import find_spline_interval, spline_x, spline_y, heading
from MPC.python_scripts.dynamics import linearize_dynamics, discrete_dynamics
from MPC.python_scripts.cost import tracking_linear_term,tracking_quadratic_term
from MPC.python_scripts.utils import end_horizon_idces

class MPCValues:
    def __init__(self, path, num_modes: int = 1, horizon: int = 40,
                 consensus_horizon: int = 4, initial_state=None,
                 initial_control=None, timestep: float = 0.25, state_dim: int = 5,
                 control_dim: int = 3, num_obstacles: int = 1, contouring_cost: float = 5.0,
                 terminal_contouring: float = 5.0,
                 lag_cost: float = 5.0, progress_reward: float = 0.01, yaw_rate_change: float = 0.01,
                 acceleration_change: float = 0.01, path_velocity_change: float = 0.01,
                 obstacle_horizon: int = 13):

        if initial_state is None:
            initial_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        if initial_control is None:
            initial_control = np.array([0.0, 0.0, 0.0])

        self.S_state = consensus_horizon + num_modes * (horizon - consensus_horizon)
        self.S_control = self.S_state - 1
        self.S_discrete = self.S_state - num_modes

        self.path = path
        self.num_modes = num_modes
        self.horizon = horizon
        self.consensus_horizon = consensus_horizon
        self.timestep = timestep
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.num_obstacles = num_obstacles
        self.initial_state = initial_state
        self.initial_control = initial_control
        self.contouring_cost = contouring_cost
        self.terminal_contouring = terminal_contouring
        self.lag_cost = lag_cost
        self.progress_reward = progress_reward
        self.yaw_rate_change = yaw_rate_change
        self.acceleration_change = acceleration_change
        self.path_velocity_change = path_velocity_change
        self.obstacle_horizon = obstacle_horizon

        self.control_penalty = np.diag(
            np.tile(np.array([yaw_rate_change, acceleration_change, path_velocity_change]), self.S_control - num_modes))
        self.progress_reward_vector = progress_reward * timestep * np.ones(self.S_control)
        self.contouring_states = [np.zeros(3) for _ in range(self.S_state)]  # cs
        self.linear_contouring = [np.zeros((3, 3)) for _ in range(self.S_state)]  # Γs
        self.contouring_state = np.zeros(3 * self.S_state)  # c
        self.linear_contouring_matrix = np.zeros((3 * self.S_state, 3 * self.S_state))  # Γ
        self.gs = [np.zeros(state_dim) for _ in range(self.S_discrete)]
        self.As = [np.zeros((state_dim, state_dim)) for _ in range(self.S_discrete)]
        self.Bs = [np.zeros((state_dim, control_dim)) for _ in range(self.S_discrete)]
        self.robot_positions = [np.zeros(2) for _ in range(self.S_state)]
        self.obstacles = dict()
        self.obstacle_indicator = [False for _ in range(num_obstacles)]


def initial_guess(vals, v0=None):
    """
    Produces an initial trajectory for the first MPC iteration.
    NOTE: Check heading and wrap to π potential issues.
    """
    path = vals.path
    n_modes = vals.num_modes
    N = vals.horizon
    k_c = vals.consensus_horizon
    dt = vals.timestep

    if v0 is None:
        v0 = vals.initial_state[3]
    s0 = vals.initial_state[4]

    s_guess = np.linspace(s0, s0 + v0 * dt * (N - 1), N)
    spline_indices = [find_spline_interval(s, path) for s in s_guess]

    X_guess = [spline_x(s, path, idx) for s, idx in zip(s_guess, spline_indices)]
    Y_guess = [spline_y(s, path, idx) for s, idx in zip(s_guess, spline_indices)]
    psi_guess = [heading(s, path, idx) for s, idx in zip(s_guess, spline_indices)]
    v_guess = v0 * np.ones(N)

    omega_guess = np.diff(psi_guess) / dt
    a_guess = np.zeros(N - 1)
    vs_guess = v0 * np.ones(N - 1)

    qs = np.vstack([X_guess, Y_guess, psi_guess, v_guess, s_guess])
    us = np.vstack([omega_guess, a_guess, vs_guess])

    qs = np.hstack([qs, np.repeat(qs[:, k_c:], n_modes - 1, axis=1)])
    us = np.hstack([us, np.repeat(us[:, k_c - 1:], n_modes - 1, axis=1)])

    return qs, us


class MPCProblem:
    def __init__(self, dynamics, vals, scene, qs, us):
        self.dynamics = dynamics
        self.vals = vals
        self.scene = scene
        self.model = ca.Opti()
        self.q = self.model.variable(vals.state_dim, vals.S_state)
        self.u = self.model.variable(vals.control_dim, vals.S_control)
        self.qs = qs
        self.us = us
        self.construct_problem()

    def construct_problem(self):
        self.state_constraints()
        self.control_constraints()
        self.dynamics_constraints()
        self.obstacle_constraints()
        self.tracking_cost_matrices()
        self.assemble_objective()

    def state_constraints(self):
        self.model.subject_to(self.vals.initial_state == self.q[:, 0])
        for k in range(self.vals.horizon):
            self.model.subject_to(self.q[3, k] <= 12)
            self.model.subject_to(self.q[3, k] >= 0)

    def control_constraints(self):
        S_u = self.vals.S_control

        omega_max = self.dynamics.u_limit.omega_max
        omega_min = self.dynamics.u_limit.omega_min
        a_max = self.dynamics.u_limit.a_max
        a_min = self.dynamics.u_limit.a_min
        vs_max = self.dynamics.u_limit.vs_max
        vs_min = self.dynamics.u_limit.vs_min

        tan_param = np.tan(np.radians(30.))
        L_param = 4.

        for k in range(S_u):
            self.model.subject_to(L_param * self.u[0, k] - tan_param * self.q[3, k] <= 0)
            self.model.subject_to(L_param * self.u[0, k] + tan_param * self.q[3, k] >= 0)
            self.model.subject_to(self.u[0, k] <= omega_max)
            self.model.subject_to(self.u[0, k] >= omega_min)
            self.model.subject_to(self.u[1, k] <= a_max)
            self.model.subject_to(self.u[1, k] >= a_min)
            self.model.subject_to(self.u[2, k] <= vs_max)
            self.model.subject_to(self.u[2, k] >= vs_min)

    def dynamics_constraints(self):
        n_modes = self.vals.num_modes
        N = self.vals.horizon
        k_c = self.vals.consensus_horizon
        dt = self.vals.timestep
        n = self.vals.state_dim
        m = self.vals.control_dim

        As, Bs, gs = [], [], []

        for k in range(k_c - 1):
            Ak, Bk = linearize_dynamics(discrete_dynamics, self.qs[0:n - 1, k], self.us[0:m - 1, k], dt)
            self.vals.As[k] = Ak
            self.vals.Bs[k] = Bk
            self.vals.gs[k] = discrete_dynamics(self.qs[0:n - 1, k], self.us[0:m - 1, k], dt) - Ak @ self.qs[0:n - 1, k]

            As.append(Ak)
            Bs.append(Bk)
            gs.append(self.vals.gs[k])

            self.model.subject_to(
                ca.mtimes(As[k], self.q[0:n - 1, k]) + ca.mtimes(Bs[k], self.u[0:m - 1, k]) + gs[k] == self.q[0:n - 1, k + 1])
            self.model.subject_to(dt * self.u[m - 1, k] + self.q[n - 1, k] == self.q[n - 1, k + 1])

        # # Constraints for transitions from k_c to subsequent time step for each mode
        # Ak_c, Bk_c = linearize_dynamics(discrete_dynamics, self.qs[0:n - 1, k_c], self.us[0:m - 1, k_c], dt)
        # self.vals.As[k_c] = Ak_c
        # self.vals.Bs[k_c] = Bk_c
        # self.vals.gs[k_c] = discrete_dynamics(self.qs[0:n - 1, k_c], self.us[0:m - 1, k_c], dt) - Ak_c @ self.qs[0:n - 1, k_c] - Bk_c @ self.us[0:m - 1, k_c]
        #
        # As.append(Ak_c)
        # Bs.append(Bk_c)
        # gs.append(self.vals.gs[k_c])
        #
        # for j in range(n_modes):
        #     next_state_idx = k_c + (j * (N - k_c))
        #     self.model.subject_to(
        #         ca.mtimes(As[k_c], self.q[0:n - 1, k_c]) + ca.mtimes(Bs[k_c], self.u[0:m - 1, k_c]) + gs[k_c] == self.q[0:n - 1, next_state_idx])
        #     self.model.subject_to(dt * self.u[m - 1, k_c] + self.q[n - 1, k_c] == self.q[n - 1, next_state_idx])
        #
        # # Constraints for transitions for each parallel horizon "tail"
        # for j in range(n_modes):
        #     idx_offset = (j * (N - k_c))
        #     start_idx = k_c + idx_offset
        #     end_idx = start_idx + (N - k_c - 1)
        #
        #     for k in range(start_idx, end_idx):
        #         Ak, Bk = linearize_dynamics(discrete_dynamics, self.qs[0:n - 1, k], self.us[0:m - 1, k], dt)
        #         self.vals.As[k] = Ak
        #         self.vals.Bs[k] = Bk
        #         self.vals.gs[k] = discrete_dynamics(self.qs[0:n - 1, k], self.us[0:m - 1, k], dt) - Ak @ self.qs[0:n - 1, k] - Bk @ self.us[0:m - 1, k]
        #
        #         As.append(Ak)
        #         Bs.append(Bk)
        #         gs.append(self.vals.gs[k])
        #
        #         self.model.subject_to(
        #             ca.mtimes(As[k], self.q[0:n - 1, k]) + ca.mtimes(Bs[k], self.u[0:m - 1, k]) + gs[k] == self.q[0:n - 1, k + 1])
        #         self.model.subject_to(dt * self.u[m - 1, k] + self.q[n - 1, k] == self.q[n - 1, k + 1])

    def obstacle_constraints(self):
            n_modes = self.vals.num_modes
            N = self.vals.horizon
            k_c = self.vals.consensus_horizon
            # n_obs = self.vals.num_obstacles
            N_obs = self.vals.obstacle_horizon
            node_ids = self.scene.node_ids
            S_q = self.vals.S_state

            ps = []
            p_obs = []
            obs_on = [self.vals.obstacles[node_id].active for node_id in node_ids]

            b = 3.0
            for k in range(S_q):
                self.vals.robot_positions[k] = self.qs[0:2, k]
                ps.append(self.vals.robot_positions[k])

            for j in range(n_modes):
                for k in range(max(k_c, N_obs)):
                    for i, node_id in enumerate(node_ids):
                        if k < k_c:
                            k_os = k
                            p_obs.append(self.vals.obstacles[node_id].positions[j][k])
                        else:
                            idx_os = j * (N - k_c)  # TODO: if something gets wrong, check here
                            k_os = k + idx_os
                            p_obs.append(self.vals.obstacles[node_id].positions[j][k])

                        a = (ps[k] - p_obs[-1]) / np.linalg.norm(ps[k] - p_obs[-1])
                        self.model.subject_to(obs_on[i] * (a.T @ (self.q[0:2, k_os] - p_obs[-1])) >= obs_on[i] * b)

            return ps, p_obs, obs_on

    def tracking_cost_matrices(self):
        S_q = self.vals.S_state
        cs = []
        Γs = []
        for k in range(S_q):
            P0 = self.qs[:3, k]  # X, Y, s of the initial guess
            spline_idx = find_spline_interval(P0[2], self.vals.path)
            if k in end_horizon_idces(self.vals):
                ck = tracking_linear_term(P0, self.vals.terminal_contouring, self.vals.lag_cost, self.vals.path, spline_idx)
                Γk = tracking_quadratic_term(P0, self.vals.terminal_contouring, self.vals.lag_cost, self.vals.path, spline_idx)
            else:
                ck = tracking_linear_term(P0, self.vals.contouring_cost, self.vals.lag_cost, self.vals.path, spline_idx)
                Γk = tracking_quadratic_term(P0, self.vals.contouring_cost, self.vals.lag_cost, self.vals.path, spline_idx)
            cs.append(ck)
            Γs.append(Γk)
        self.vals.contouring_state = np.concatenate(cs)
        self.vals.linear_contouring_matrix = block_diag(*Γs)

    def assemble_objective(self):
        n_modes = self.vals.num_modes
        N = self.vals.horizon
        k_c = self.vals.consensus_horizon
        dt = self.vals.timestep
        R = self.vals.control_penalty
        L = self.vals.progress_reward
        u0 = self.vals.initial_state

        # Tracking error
        tracking_error = 0
        for k in range(self.vals.S_state):
            P = self.q[:3, k]
            tracking_error += ca.mtimes([P.T, self.vals.Γ[k], P]) - 2 * ca.mtimes([P.T, self.vals.c[k]])

        # Control effort
        Δu = self.u[:, 1:] - self.u[:, :-1]
        control_effort = ca.mtimes([Δu.reshape((-1, 1)).T, R, Δu.reshape((-1, 1))])

        # Input rate cost
        r = np.array([self.vals.yaw_rate_change, self.vals.acceleration_change, self.vals.path_velocity_change])
        input_rate_cost = ca.mtimes([(self.u[:, 0] - u0).T, ca.diag(r), (self.u[:, 0] - u0)])

        # Progress reward
        m = self.vals.control_dim
        progress_reward = ca.mtimes(self.vals.progress_reward.T, self.u[m - 1, :])

        # Assemble objective
        objective = tracking_error + control_effort + input_rate_cost - progress_reward

        self.model.minimize(objective)

    def solve(self):
        # Solve the optimization problem here
        p_opts = {"expand": False}
        s_opts = {"max_iter": 1000}
        self.model.solver('ipopt', p_opts, s_opts)
        result = self.model.solve()
        return result.value(self.q), result.value(self.u)


def update_problem(vals, qs, us):

    pass
