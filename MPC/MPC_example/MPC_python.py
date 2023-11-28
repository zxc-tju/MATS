import sys
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap

from MPC.python_scripts.utils import (load_model, load_data_set, predicted_dynamics, get_recorded_robot_controls,
                                      update_obstacles_from_predictions, predict_future_states)
from MPC.python_scripts.structs import (get_scene_info, ControlLimits, DynamicsModel,
                                        PredictionSettings, init_node_obstacles)
from MPC.python_scripts.path_handling import load_splines, SplinePath, find_best_s
from MPC.python_scripts.mpc import MPCProblem, MPCValues, initial_guess
import MPC.python_scripts.plot as plotting_helper

sys.path.append("../../mats")
from utils import prediction_output_to_trajectories


" ==== Prepare model and data ==== "
# load data
env = load_data_set("/home/zxc/codes/MATS/experiments/processed/nuScenes_val_full.pkl")

# load model
mats, hyperparams = load_model('../../experiments/nuScenes/models'
                               '/models_26_Jul_2020_17_11_34_full_zeroRrows_batch8_fixed_edges',
                               env, ts=16)

mats.set_environment(env)
mats.set_annealing_params()
ph = hyperparams['prediction_horizon']
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
# nuScenes_data_path = '/home/zxc/codes/MATS/experiments/nuScenes/data'  # for home
nuScenes_data_path = '/home/zxc/Downloads/nuscene'  # for 423
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
t_range = range(2, 10)

" ==== Select Scene ==== "
# select scene
scene_num = 23  # corresponds to 24 in julia indexing
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
x_min = x_center - 75.0
y_min = y_center - 75.0
x_max = x_center + 75.0
y_max = y_center + 75.0

my_patch = (x_min, y_min, x_max, y_max)

" ==== MPC settings ==== "
# model prediction settings
num_modes = 1
pred_settings = PredictionSettings(mats, hyperparams, env, num_modes)

# get path data
x_coefs_var, y_coefs_var, breaks_var = load_splines()
path_obj = SplinePath(x_coefs_var, y_coefs_var, breaks_var)

# MPC parameters, constraints and settings
control_limits_obj = ControlLimits(0.7, -0.7, 4.0, -5.0, 12.0, 0.0)
dynamics_obj = DynamicsModel(4, 2, control_limits_obj)

" ==== MPC process ==== "

" --- Initialize MPC --- "
for first_ts in tqdm(t_range):

    # robot initial state
    q0 = [robot_node.x[first_ts], robot_node.y[first_ts], robot_node.theta[first_ts], robot_node.v[first_ts], 0]
    q0[4] = find_best_s(q0, path_obj, enable_global_search=True)

    vals_obj = MPCValues(path_obj, initial_state=q0,
                         num_modes=num_modes, horizon=13,
                         consensus_horizon=4, num_obstacles=len(non_robot_nodes), timestep=scene.dt)

    # make first predictions for obstacle constraints
    init_node_obstacles(non_robot_node_ids, vals_obj)
    Aps, Bps, gps, q_pred0, nodes_present, mats_outputs = predicted_dynamics(pred_settings, scene_num, first_ts)
    u_pred = get_recorded_robot_controls(pred_settings, scene_num, first_ts)
    q_pred = [predict_future_states(pred_settings, q_pred0, u_pred, Aps, Bps, gps, j) for j in
              range(0, vals_obj.num_modes)]
    update_obstacles_from_predictions(q_pred, nodes_present, vals_obj, scene)

    # save prediction results
    mats_outputs_collection.append(mats_outputs)

    # initial solution guess
    initial_state, initial_control = initial_guess(vals_obj)

    # construct problem
    mpc = MPCProblem(dynamics_obj, vals_obj, non_robot_node_ids, initial_state, initial_control)

    " ==== Solve MPC ==== "
    time_start = time.time()
    state_star, control_star = mpc.solve()
    print('output states', state_star[:, 0:3])
    print('time consumption: ', time.time() - time_start)

    robot_state_collection.append(state_star)


" ==== Visualize Results ==== "
# Plot predicted timestep
plotting_helper.plot_multi_frame_dist(my_patch, layers, mats_outputs_collection, scene, nusc_map=nusc_map,
                                      max_hl=max_hl, ph=ph, x_min=scene.x_min,
                                      y_min=scene.y_min,
                                      robot_plan=robot_state_collection, scene_num=scene_num)

