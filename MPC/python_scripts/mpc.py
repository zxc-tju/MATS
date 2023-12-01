import numpy as np
import casadi as ca
from casadi import mtimes, MX
from scipy.linalg import block_diag

from MPC.python_scripts.path_handling import find_spline_interval, spline_x, spline_y, heading, find_best_s
from MPC.python_scripts.dynamics import linearize_dynamics, discrete_dynamics, simulate_path_dynamics
from MPC.python_scripts.cost import tracking_linear_term, tracking_quadratic_term
from MPC.python_scripts.utils import end_horizon_idces


class MPCValues:
    def __init__(self, path, num_modes: int = 1, horizon: int = 40,
                 consensus_horizon: int = 4, initial_state=None,
                 initial_control=None, timestep: float = 0.25, state_dim: int = 5,
                 control_dim: int = 3, num_obstacles: int = 1, contouring_cost: float = 5.0,
                 terminal_contouring: float = 5.0,
                 lag_cost: float = 5.0, progress_reward: float = 0.01, yaw_rate_change: float = 0.01,
                 acceleration_change: float = 0.01, path_velocity_change: float = 0.01,
                 obstacle_horizon: int = 7):

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
        self.gs = [np.zeros(state_dim) for _ in range(self.S_discrete)]  # linearization point
        self.As = [np.zeros((state_dim, state_dim)) for _ in range(self.S_discrete)]  # dynamics matrices
        self.Bs = [np.zeros((state_dim, control_dim)) for _ in range(self.S_discrete)]  # input matrices
        self.robot_positions = [np.zeros(2) for _ in range(self.S_state)]
        self.obstacles = dict()
        self.obstacle_indicator = [False for _ in range(num_obstacles)]

    def update_problem(self, initial_state_plan, initial_control_plan):
        path = self.path
        num_modes = self.num_modes
        horizon = self.horizon
        consensus_horizon = self.consensus_horizon
        state_dim = self.state_dim
        control_dim = self.control_dim
        timestep = self.timestep
        contouring_cost = self.contouring_cost
        terminal_contouring = self.terminal_contouring
        lag_cost = self.lag_cost

        S_state = self.S_state

        # Update initial state and prev control
        self.initial_state = initial_state_plan[:, 1]
        self.initial_state[4] = find_best_s(self.initial_state, path)
        self.initial_control = initial_control_plan[:, 0]

        # Shift states and controls over consensus portion
        initial_state_plan[:, :consensus_horizon] = initial_state_plan[:, 1:consensus_horizon + 1]
        initial_control_plan[:, :consensus_horizon - 1] = initial_control_plan[:, 1:consensus_horizon]

        # Shift dynamics and tracking error matrices over consensus portion
        self.As[:consensus_horizon] = self.As[1:consensus_horizon + 1]
        self.Bs[:consensus_horizon] = self.Bs[1:consensus_horizon + 1]
        self.gs[:consensus_horizon] = self.gs[1:consensus_horizon + 1]

        self.contouring_states[:consensus_horizon] = self.contouring_states[1:consensus_horizon + 1]
        self.linear_contouring[:consensus_horizon] = self.linear_contouring[1:consensus_horizon + 1]

        for j in range(1, num_modes + 1):
            horizon_start = consensus_horizon + (j - 1) * (horizon - consensus_horizon)
            horizon_end = horizon_start + horizon - consensus_horizon - 1

            # Shift states and controls for each horizon
            initial_state_plan[:, horizon_start:horizon_end - 1] = initial_state_plan[:, horizon_start + 1:horizon_end]
            initial_control_plan[:, horizon_start:horizon_end - 2] = initial_control_plan[:,
                                                                     horizon_start + 1:horizon_end - 1]

            # Simulate for new last state for each horizon
            q_last_A = discrete_dynamics(initial_state_plan[:state_dim - 1, horizon_end],
                                         initial_control_plan[:control_dim - 1, horizon_end - 1], timestep)
            q_last_B = simulate_path_dynamics(initial_state_plan[-1, horizon_end],
                                              initial_control_plan[-1, horizon_end - 1])
            q_last = np.concatenate([q_last_A, np.array([q_last_B])])
            initial_state_plan[:, horizon_end] = q_last

            # Shift dynamics matrices for each horizon
            start_idx = horizon_start
            end_idx = horizon_start + horizon - consensus_horizon - 2 - (j - 1)

            self.As[start_idx:end_idx - 1] = self.As[start_idx + 1:end_idx]
            self.Bs[start_idx:end_idx - 1] = self.Bs[start_idx + 1:end_idx]
            self.gs[start_idx:end_idx - 1] = self.gs[start_idx + 1:end_idx]

            # Fill in last entry of dynamics matrices for each horizon
            q_last = initial_state_plan[:, horizon_end - 2]
            u_last = initial_control_plan[:, horizon_end - 2]
            A_last, B_last = linearize_dynamics(discrete_dynamics, initial_state_plan[:state_dim - 1, horizon_end - 1],
                                                initial_control_plan[:control_dim - 1, horizon_end - 1], timestep)
            self.As[end_idx], self.Bs[end_idx] = A_last, B_last
            self.gs[end_idx] = discrete_dynamics(initial_state_plan[:state_dim - 1, horizon_end - 1],
                                                 initial_control_plan[:control_dim - 1, horizon_end - 1],
                                                 timestep) - A_last @ initial_state_plan[:state_dim - 1,
                                                                      horizon_end - 1] - B_last @ initial_control_plan[
                                                                                                  :control_dim - 1,
                                                                                                  horizon_end - 1]

            # Shift tracking cost matrices for each horizon
            self.contouring_states[horizon_start:horizon_end - 1] = self.contouring_states[
                                                                    horizon_start + 1:horizon_end]
            self.linear_contouring[horizon_start:horizon_end - 1] = self.linear_contouring[
                                                                    horizon_start + 1:horizon_end]

            # Correct second to last so that the weight is qc rather than terminal_contouring
            P_stl = [initial_state_plan[0, horizon_end - 1], initial_state_plan[1, horizon_end - 1],
                     initial_state_plan[4, horizon_end - 1]]  # X, Y, s of last state of previous solution
            spline_idx = find_spline_interval(P_stl[2], path)
            c_stl = tracking_linear_term(P_stl, contouring_cost, lag_cost, path, spline_idx)
            Γ_stl = tracking_quadratic_term(P_stl, contouring_cost, lag_cost, path, spline_idx)
            self.contouring_states[horizon_end - 1] = c_stl
            self.linear_contouring[horizon_end - 1] = Γ_stl

            # Update last entry
            P_last = [q_last[0], q_last[1], q_last[4]]  # X, Y, s of new last state appended
            spline_idx = find_spline_interval(P_last[2], path)
            c_last = tracking_linear_term(P_last, terminal_contouring, lag_cost, path, spline_idx)
            Γ_last = tracking_quadratic_term(P_last, terminal_contouring, lag_cost, path, spline_idx)
            self.contouring_states[horizon_end] = c_last
            self.linear_contouring[horizon_end] = Γ_last

        self.contouring_state = np.concatenate(self.contouring_states)
        self.linear_contouring_matrix = block_diag(*self.linear_contouring)

        # Update positions for obstacle constraints
        for k in range(S_state):
            self.robot_positions[k] = initial_state_plan[:2, k]

        return initial_state_plan, initial_control_plan


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
    def __init__(self, dynamics, vals, non_robot_node_ids, q_initial, u_initial):
        self.dynamics = dynamics
        self.vals = vals
        self.non_robot_node_ids = non_robot_node_ids
        self.model = ca.Opti()
        self.q = self.model.variable(vals.state_dim, vals.S_state)
        self.u = self.model.variable(vals.control_dim, vals.S_control)
        self.q_initial = q_initial  # [x, y, heading, v, s]
        self.u_initial = u_initial
        self.construct_problem()

    def construct_problem(self):
        self.state_constraints()
        self.control_constraints()
        self.dynamics_constraints()
        self.obstacle_constraints()
        self.tracking_cost_matrices()
        self.assemble_objective()

    def update_problem(self, vals, q_initial, u_initial):

        self.construct_problem()

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
        horizon = self.vals.horizon
        consensus_horizon = self.vals.consensus_horizon
        dt = self.vals.timestep
        state_dim = self.vals.state_dim
        control_dim = self.vals.control_dim

        As, Bs, gs = [], [], []

        for k in range(consensus_horizon - 1):
            Ak, Bk = linearize_dynamics(discrete_dynamics, self.q_initial[0:state_dim - 1, k],
                                        self.u_initial[0:control_dim - 1, k], dt)
            self.vals.As[k] = Ak
            self.vals.Bs[k] = Bk
            self.vals.gs[k] = discrete_dynamics(self.q_initial[0:state_dim - 1, k],
                                                self.u_initial[0:control_dim - 1, k], dt) - Ak @ self.q_initial[
                                                                                                 0:state_dim - 1, k]

            As.append(Ak)
            Bs.append(Bk)
            gs.append(self.vals.gs[k])

            self.model.subject_to(
                ca.mtimes(As[k], self.q[0:state_dim - 1, k]) + ca.mtimes(Bs[k], self.u[0:control_dim - 1, k]) + gs[
                    k] == self.q[0:state_dim - 1, k + 1])
            self.model.subject_to(
                dt * self.u[control_dim - 1, k] + self.q[state_dim - 1, k] == self.q[state_dim - 1, k + 1])

        # Constraints for transitions from k_c to subsequent time step for each mode
        Ak_c, Bk_c = linearize_dynamics(discrete_dynamics, self.q_initial[0:state_dim - 1, consensus_horizon - 1],
                                        self.u_initial[0:control_dim - 1, consensus_horizon - 1], dt)
        self.vals.As[consensus_horizon - 1] = Ak_c
        self.vals.Bs[consensus_horizon - 1] = Bk_c
        self.vals.gs[consensus_horizon - 1] = discrete_dynamics(self.q_initial[0:state_dim - 1, consensus_horizon - 1],
                                                                self.u_initial[0:control_dim - 1,
                                                                consensus_horizon - 1], dt) - Ak_c @ self.q_initial[
                                                                                                     0:state_dim - 1,
                                                                                                     consensus_horizon - 1] - Bk_c @ self.u_initial[
                                                                                                                                     0:control_dim - 1,
                                                                                                                                     consensus_horizon - 1]

        As.append(Ak_c)
        Bs.append(Bk_c)
        gs.append(self.vals.gs[consensus_horizon - 1])

        for j in range(n_modes):
            next_state_idx = consensus_horizon + (j * (horizon - consensus_horizon))
            self.model.subject_to(
                ca.mtimes(As[consensus_horizon - 1], self.q[0:state_dim - 1, consensus_horizon - 1]) + ca.mtimes(
                    Bs[consensus_horizon - 1], self.u[0:control_dim - 1, consensus_horizon - 1]) + gs[
                    consensus_horizon - 1] == self.q[0:state_dim - 1, next_state_idx])
            self.model.subject_to(
                dt * self.u[control_dim - 1, consensus_horizon - 1] + self.q[state_dim - 1, consensus_horizon - 1] ==
                self.q[state_dim - 1, next_state_idx])

        # Constraints for transitions for each parallel horizon "tail"
        for j in range(n_modes):
            idx_offset = j * (horizon - consensus_horizon)
            start_idx = consensus_horizon + idx_offset
            end_idx = start_idx + (horizon - consensus_horizon - 1)

            for k in range(start_idx, end_idx):
                Ak, Bk = linearize_dynamics(discrete_dynamics, self.q_initial[0:state_dim - 1, k],
                                            self.u_initial[0:control_dim - 1, k], dt)
                self.vals.As[k] = Ak
                self.vals.Bs[k] = Bk
                self.vals.gs[k] = discrete_dynamics(self.q_initial[0:state_dim - 1, k],
                                                    self.u_initial[0:control_dim - 1, k], dt) - Ak @ self.q_initial[
                                                                                                     0:state_dim - 1,
                                                                                                     k] - Bk @ self.u_initial[
                                                                                                               0:control_dim - 1,
                                                                                                               k]

                As.append(Ak)
                Bs.append(Bk)
                gs.append(self.vals.gs[k])

                self.model.subject_to(
                    ca.mtimes(As[k], self.q[0:state_dim - 1, k]) + ca.mtimes(Bs[k], self.u[0:control_dim - 1, k]) + gs[
                        k] == self.q[0:state_dim - 1, k + 1])
                self.model.subject_to(
                    dt * self.u[control_dim - 1, k] + self.q[state_dim - 1, k] == self.q[state_dim - 1, k + 1])

    def obstacle_constraints(self):
        n_modes = self.vals.num_modes
        horizon = self.vals.horizon
        consensus_horizon = self.vals.consensus_horizon
        # n_obs = self.vals.num_obstacles
        obstacle_horizon = self.vals.obstacle_horizon
        node_ids = self.non_robot_node_ids
        S_q = self.vals.S_state

        robot_position = []
        obstacle_position = []
        obs_on = [self.vals.obstacles[node_id].active for node_id in node_ids]

        b = 3
        for t in range(S_q):
            self.vals.robot_positions[t] = self.q_initial[0:2, t]
            robot_position.append(self.vals.robot_positions[t])

        for j in range(n_modes):  # n_modes=1
            for t in range(max(consensus_horizon, obstacle_horizon)):
                for i, node_id in enumerate(node_ids):
                    if t < consensus_horizon:
                        t_offset = t
                        obstacle_position.append(self.vals.obstacles[node_id].positions[j][t])
                    else:
                        idx_offset = j * (horizon - consensus_horizon)  # idx_offset is always zero if n_modes=1
                        t_offset = t + idx_offset
                        obstacle_position.append(self.vals.obstacles[node_id].positions[j][t])
                    if obs_on[i]:
                        a = (robot_position[t_offset] - obstacle_position[-1]) / np.linalg.norm(
                            robot_position[t_offset] - obstacle_position[-1])
                        a_mx = MX(a).T  # Convert 'a' from NumPy to MX type
                        self.model.subject_to(a_mx @ (self.q[0:2, t_offset] - obstacle_position[-1]) >= b)

        # return ps, obstacle_position, obs_on

    def tracking_cost_matrices(self):
        S_q = self.vals.S_state
        cs = []
        Γs = []
        for k in range(S_q):
            P0 = [self.q_initial[0, k], self.q_initial[1, k], self.q_initial[4, k]]  # X, Y, s of the initial guess
            spline_idx = find_spline_interval(P0[2], self.vals.path)
            if k in end_horizon_idces(self.vals):
                ck = tracking_linear_term(P0, self.vals.terminal_contouring, self.vals.lag_cost, self.vals.path,
                                          spline_idx)
                Γk = tracking_quadratic_term(P0, self.vals.terminal_contouring, self.vals.lag_cost, self.vals.path,
                                             spline_idx)
            else:
                ck = tracking_linear_term(P0, self.vals.contouring_cost, self.vals.lag_cost, self.vals.path, spline_idx)
                Γk = tracking_quadratic_term(P0, self.vals.contouring_cost, self.vals.lag_cost, self.vals.path,
                                             spline_idx)
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
        u0 = self.vals.initial_control

        # Tracking error
        P_array = MX(self.q[[0, 1, 4], :])  # X, Y, s of the initial guess
        P = P_array.reshape((-1, 1))
        tracking_error = ca.mtimes([P.T, self.vals.linear_contouring_matrix, P]) + ca.mtimes(
            [P.T, self.vals.contouring_state])

        # Control effort
        Δu = self.u[:, 1:] - self.u[:, :-1]
        control_effort = ca.mtimes([Δu.reshape((-1, 1)).T, R, Δu.reshape((-1, 1))])

        # Input rate cost
        r = np.array([self.vals.yaw_rate_change, self.vals.acceleration_change, self.vals.path_velocity_change])
        input_rate_cost = ca.mtimes([(self.u[:, 0] - u0).T, ca.diag(r), (self.u[:, 0] - u0)])

        # Progress reward
        m = self.vals.control_dim
        progress_reward = ca.mtimes(self.u[m - 1, :], self.vals.progress_reward_vector)

        # Assemble objective
        objective = tracking_error + control_effort + input_rate_cost - progress_reward

        self.model.minimize(objective)

    def solve(self):
        # Solve the optimization problem here
        p_opts = {"expand": True}
        s_opts = {"max_iter": 1000, "print_level": 0}
        self.model.solver('ipopt', p_opts, s_opts)
        try:
            # Attempt to solve the optimization problem
            result = self.model.solve()

            return result.value(self.q), result.value(self.u)

        except Exception as e:  # Catching all exceptions can help with debugging
            print("An exception occurred: ", str(e))
            q_val = self.model.debug.value(self.q)
            u_val = self.model.debug.value(self.u)
            print("q values at exception: ", q_val)
            print("u values at exception: ", u_val)

            # Retrieve the number of constraints
            num_constraints = self.model.g.numel()
            # Loop over each constraint and print its structure and value
            for i in range(num_constraints):
                # Get the symbolic representation of the constraint
                con = self.model.g[i]
                # Evaluate the constraint value using debug at the current iterate
                con_value = self.model.debug.value(con)
                print(f"Constraint {i}: {con} value at current iterate: {con_value}")

            return None, None
