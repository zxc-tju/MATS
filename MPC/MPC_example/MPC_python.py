import sys
import os
import time

sys.path.append("../../mats")
from utils import prediction_output_to_trajectories
from tqdm import tqdm, trange
from model.model_registrar import ModelRegistrar
from model.mats import MATS
from environment.node import MultiNode
from model.dataset import EnvironmentDataset
import evaluation
import visualization
# import helper as plotting_helper


from MPC.python_scripts.utils import *
from MPC.python_scripts.structs import *
from MPC.python_scripts.path_handling import *
from MPC.python_scripts.mpc import *

# load data
env = load_data_set("/home/zxc/codes/MATS/experiments/processed/nuScenes_val_full.pkl")

# load model
model_path = "../../experiments/nuScenes/models/models_21_Jul_2020_10_25_10_full_zeroRrows_batch8_double_fixed_a_norm"
mats, hyperparams = load_model(model_path, env, ts=11)

# model prediction settings
num_modes = 1
pred_settings = PredictionSettings(mats, hyperparams, env, num_modes)
# select scene
scene_num = 23  # corresponds to 24 in julia indexing
scene = create_scene(env, scene_num)
# print(scene.robot.x[2:10])

if not os.path.isdir('./data'):
    os.mkdir('./data')

# select time interval
first_ts = 2
last_ts = scene.timesteps

# get path data
x_coefs_var, y_coefs_var, breaks_var = load_splines()
path_obj = SplinePath(x_coefs_var, y_coefs_var, breaks_var)

# robot initial state
q0 = [scene.robot.x[first_ts], scene.robot.y[first_ts], scene.robot.theta[first_ts], scene.robot.v[first_ts], 0]
q0[4] = find_best_s(q0, path_obj, enable_global_search=True)

# MPC parameters, constraints and settings
control_limits_obj = ControlLimits(0.7, -0.7, 4.0, -5.0, 12.0, 0.0)
dynamics_obj = DynamicsModel(4, 2, control_limits_obj)
vals_obj = MPCValues(path_obj, num_modes=num_modes, horizon=25,
                     consensus_horizon=4, initial_state=q0, num_obstacles=len(scene.non_robot_nodes))

# make first predictions for obstacle constraints
init_node_obstacles(scene, vals_obj)
Aps, Bps, gps, q_pred0, nodes_present = predicted_dynamics(pred_settings, scene_num, first_ts)
u_pred = get_recorded_robot_controls(pred_settings, scene_num, first_ts)
q_pred = [predict_future_states(pred_settings, q_pred0, u_pred, Aps, Bps, gps, j) for j in range(0, vals_obj.num_modes)]
update_obstacles_from_predictions(q_pred, nodes_present, vals_obj, scene)

# initial solution guess
qs, us = initial_guess(vals_obj)

# construct problem
mpc = MPCProblem(dynamics_obj, vals_obj, scene, qs, us)

q_star, u_star = mpc.solve()
print('output q', q_star[:, 0:3])
# print(u_star)


"Solve MPC"
