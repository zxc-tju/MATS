import sys
import os
import time
from tqdm import tqdm, trange
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap

from MPC.python_scripts.utils import (load_model, load_data_set, predicted_dynamics, get_recorded_robot_controls,
                                      update_obstacles_from_predictions, predict_future_states)
from MPC.python_scripts.structs import (get_scene_info, ControlLimits, DynamicsModel,
                                        PredictionSettings, init_node_obstacles)
from MPC.python_scripts.path_handling import load_splines, SplinePath, find_best_s, get_path_obj
from MPC.python_scripts.mpc import MPCProblem, MPCValues, initial_guess
import MPC.python_scripts.plot as plotting_helper

sys.path.append("../../mats")
from utils import prediction_output_to_trajectories


" ==== Prepare model and data ==== "
# load data
env = load_data_set("/home/zxc/codes/MATS/experiments/processed/nuScenes_val_full_doubled.pkl")

# load model
mats, hyperparams = load_model('../../experiments/nuScenes/models'
                               '/models_26_Jul_2020_17_11_34_full_zeroRrows_batch8_fixed_edges',
                               env, ts=16)

mats.set_environment(env)
mats.set_annealing_params()
prediction_horizon = hyperparams['prediction_horizon']
max_hl = hyperparams['maximum_history_length']

for attention_radius_override in hyperparams['override_attention_radius']:
    node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
    env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

if env.robot_type is None and hyperparams['incl_robot_node']:
    env.robot_type = env.NodeType[15]  # TODO: Make more general, allow the user to specify?
    for scene in env.scenes:
        scene.add_robot_from_nodes(env.robot_type,
                                   hyperparams=hyperparams,
                                   min_timesteps=hyperparams['minimum_history_length'] + 1 + hyperparams[
                                       'prediction_horizon'])

scenes = env.scenes

# Data for plots
nuScenes_data_path = '/home/zxc/codes/MATS/experiments/nuScenes/data'  # for home
# nuScenes_data_path = '/home/zxc/Downloads/nuscene'  # for 423
layers = ['drivable_area',
          'road_segment',
          'lane',
          'ped_crossing',
          'walkway',
          'stop_line',
          'road_divider',
          'lane_divider']

if not os.path.isdir('./data'):
    os.mkdir('./data')

if not os.path.isdir('./plots'):
    os.mkdir('./plots')

" ==== Prepare Data Recording ==== "
mats_outputs_collection = []
robot_state_collection = []
t_range = range(2, 30)

" ==== Select Scene ==== "
# select scene
scene_num = 23  # original demo is 23
scene = scenes[scene_num]
scene.calculate_scene_graph(env.attention_radius,
                            hyperparams['edge_addition_filter'],
                            hyperparams['edge_removal_filter'])

# Prepare nuscenes map for selected scene
nusc = NuScenes(version='v1.0-trainval', dataroot=nuScenes_data_path, verbose=True)
helper = PredictHelper(nusc)
nusc_map = NuScenesMap(dataroot=nuScenes_data_path, map_name=helper.get_map_name_from_sample_token(scene.name))

robot_node, non_robot_nodes, non_robot_node_ids = get_scene_info(env, scene_num)

# Configure figure center
x_center = robot_node.x[t_range[0]]
y_center = robot_node.y[t_range[0]]
x_min = x_center - 25.0
y_min = y_center - 55.0
x_max = x_center + 75.0
y_max = y_center + 55.0

my_patch = (x_min, y_min, x_max, y_max)

" ==== MPC settings ==== "
# model prediction settings
num_modes = 1
pred_settings = PredictionSettings(mats, hyperparams, env, num_modes)

# get path data from files
# ----- archived old version used in julia ----- #
# x_coefs_var, y_coefs_var, breaks_var = load_splines()
# path_obj = SplinePath(x_coefs_var, y_coefs_var, breaks_var)
# ----- new version generated from robot trajectory ----- #
path_obj = get_path_obj(robot_node.x, robot_node.y)


# MPC parameters, constraints and settings
control_limits_obj = ControlLimits(0.7, -0.7, 4.0, -5.0, 12.0, 0.0)
dynamics_obj = DynamicsModel(4, 2, control_limits_obj)
iteration_num = 1

" ==== MPC process ==== "

" --- Initialize MPC --- "
first_ts = t_range[0]
state_star = []
# robot initial state
q0 = [robot_node.x[first_ts], robot_node.y[first_ts], robot_node.theta[first_ts], robot_node.v[first_ts], 0]
q0[4] = find_best_s(q0, path_obj, enable_global_search=True)

mpc_vals_obj = MPCValues(path_obj, initial_state=q0,
                         num_modes=num_modes, horizon=13,
                         consensus_horizon=4, num_obstacles=len(non_robot_nodes), timestep=scene.dt)

# make first predictions for obstacle constraints
init_node_obstacles(non_robot_node_ids, mpc_vals_obj)
Aps, Bps, gps, q_pred0, nodes_present, mats_outputs = predicted_dynamics(pred_settings, q0[:4], scene_num, first_ts)
u_pred = get_recorded_robot_controls(pred_settings, scene_num, first_ts)
q_pred = [predict_future_states(pred_settings, q_pred0, u_pred, Aps, Bps, gps, j) for j in
          range(0, mpc_vals_obj.num_modes)]

update_obstacles_from_predictions(q_pred, nodes_present, mpc_vals_obj, scene)

# save prediction results
mats_outputs_collection.append(mats_outputs)

# initial solution guess
initial_state_plan, initial_control_plan = initial_guess(mpc_vals_obj)

" --- Solve MPC --- "
for ts in tqdm(t_range[1:]):

    for i in range(iteration_num):
        time_start = time.time()

        # construct MPC
        mpc = MPCProblem(dynamics_obj, mpc_vals_obj, non_robot_node_ids, initial_state_plan, initial_control_plan)

        # solve problem
        state_star, control_star = mpc.solve()  # state_star include current state
        print(f'=====Results of {i} Iteration=====')
        print('output states', state_star[-1, 0:3])

        # update initial solution for MPC
        initial_state_plan = state_star.copy()
        initial_control_plan = control_star.copy()

        # update self dynamics with new plan
        mpc_vals_obj.re_linearize_dynamics(initial_state_plan, initial_control_plan)

        # update prediction on obstacles (without map offset)
        u_pred = initial_control_plan[0:2, 0:prediction_horizon]  # use control from optimization solution
        u_pred = u_pred.T
        q_pred = [predict_future_states(pred_settings, q_pred0, u_pred, Aps, Bps, gps, j) for j in
                  range(0, mpc_vals_obj.num_modes)]
        # align obstacles' positions with map offset
        update_obstacles_from_predictions(q_pred, nodes_present, mpc_vals_obj, scene)

        print('time consumption: ', time.time() - time_start)

    # save plan
    robot_state_collection.append(state_star.copy())

    # update prediction on system dynamics in the new step ts
    Aps, Bps, gps, q_pred0, nodes_present, mats_outputs = predicted_dynamics(pred_settings, initial_state_plan[:4, 1],
                                                                             scene_num, ts)
    # save prediction results
    mats_outputs_collection.append(mats_outputs)

    # prediction on obstacles (without map offset)
    u_pred = initial_control_plan[0:2, 0:prediction_horizon]  # use control from optimization solution
    u_pred = u_pred.T
    q_pred = [predict_future_states(pred_settings, q_pred0, u_pred, Aps, Bps, gps, j) for j in
              range(0, mpc_vals_obj.num_modes)]
    # align obstacles' positions with map offset
    update_obstacles_from_predictions(q_pred, nodes_present, mpc_vals_obj, scene)

    # shift [states, controls, dynamics] to the next time step
    mpc_vals_obj.update_problem(initial_state_plan, initial_control_plan)


" ==== Visualize Results ==== "
# Plot predicted timestep
plotting_helper.plot_multi_frame_dist(my_patch, layers, mats_outputs_collection, scene, nusc_map=nusc_map,
                                      max_hl=max_hl, ph=prediction_horizon, x_min=scene.x_min,
                                      y_min=scene.y_min,
                                      robot_plan=robot_state_collection, scene_num=scene_num)

