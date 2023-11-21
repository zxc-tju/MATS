import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.append("../../mats")
from utils import prediction_output_to_trajectories
from tqdm import tqdm, trange
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap

from MPC.python_scripts.utils import *
from MPC.python_scripts.structs import *
from MPC.python_scripts.path_handling import *
from MPC.python_scripts.mpc import *
import MPC.python_scripts.plot as plotting_helper


" ==== Prepare model and data ==== "
# load data
env = load_data_set("/home/zxc/codes/MATS/experiments/processed/nuScenes_val_full.pkl")

# load model
# model_path = "../../experiments/nuScenes/models/models_21_Jul_2020_10_25_10_full_zeroRrows_batch8_double_fixed_a_norm"
# mats, hyperparams = load_model(model_path, env, ts=11)

mats, hyperparams = load_model('../../experiments/nuScenes/models'
                               '/models_26_Jul_2020_17_11_34_full_zeroRrows_batch8_fixed_edges',
                               env, ts=16)

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

mats.set_environment(env)
mats.set_annealing_params()

scenes = env.scenes

ph = hyperparams['prediction_horizon']
max_hl = hyperparams['maximum_history_length']


if not os.path.isdir('./data'):
    os.mkdir('./data')

if not os.path.isdir('./plots'):
    os.mkdir('./plots')

" ==== Select Scene ==== "
# select scene
scene_num = 23  # corresponds to 24 in julia indexing
scene = scenes[scene_num]
scene.calculate_scene_graph(env.attention_radius,
                            hyperparams['edge_addition_filter'],
                            hyperparams['edge_removal_filter'])

robot_node, non_robot_nodes, non_robot_node_ids = get_scene_info(env, scene_num)

# Data for plots
nuScenes_data_path = '/home/zxc/codes/MATS/experiments/nuScenes/data'
nusc = NuScenes(version='v1.0-trainval', dataroot=nuScenes_data_path, verbose=True)
helper = PredictHelper(nusc)
nusc_map = NuScenesMap(dataroot=nuScenes_data_path, map_name=helper.get_map_name_from_sample_token(scene.name))

" ==== MPC settings ==== "
# # select time interval
# first_ts = 2
# # last_ts = scene.timesteps

for first_ts in tqdm(range(2, 10)):
    # model prediction settings
    num_modes = 1
    pred_settings = PredictionSettings(mats, hyperparams, env, num_modes)

    # get path data
    x_coefs_var, y_coefs_var, breaks_var = load_splines()
    path_obj = SplinePath(x_coefs_var, y_coefs_var, breaks_var)

    # robot initial state
    q0 = [robot_node.x[first_ts], robot_node.y[first_ts], robot_node.theta[first_ts], robot_node.v[first_ts], 0]
    q0[4] = find_best_s(q0, path_obj, enable_global_search=True)

    # MPC parameters, constraints and settings
    control_limits_obj = ControlLimits(0.7, -0.7, 4.0, -5.0, 12.0, 0.0)
    dynamics_obj = DynamicsModel(4, 2, control_limits_obj)
    vals_obj = MPCValues(path_obj, num_modes=num_modes, horizon=25,
                         consensus_horizon=4, initial_state=q0, num_obstacles=len(non_robot_nodes))

    # make first predictions for obstacle constraints
    init_node_obstacles(non_robot_node_ids, vals_obj)
    Aps, Bps, gps, q_pred0, nodes_present, mats_outputs = predicted_dynamics(pred_settings, scene_num, first_ts)
    u_pred = get_recorded_robot_controls(pred_settings, scene_num, first_ts)
    q_pred = [predict_future_states(pred_settings, q_pred0, u_pred, Aps, Bps, gps, j) for j in range(0, vals_obj.num_modes)]
    update_obstacles_from_predictions(q_pred, nodes_present, vals_obj, scene)

    # initial solution guess
    qs, us = initial_guess(vals_obj)

    # construct problem
    mpc = MPCProblem(dynamics_obj, vals_obj, non_robot_node_ids, qs, us)

    " ==== Solve MPC ==== "
    time_start = time.time()
    q_star, u_star = mpc.solve()
    print('output q', q_star[:, 0:3])
    print('time consumption: ', time.time() - time_start)
    # print(u_star)

    " ==== Visualize Results ==== "

    pred_dists, non_rob_rows, As, Bs, Qs, affine_terms, state_lengths_in_order = mats_outputs
    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(pred_dists,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=None)

    ts_key = list(prediction_dict.keys())[0]
    histories_dict = histories_dict[ts_key]
    node = list(histories_dict.keys())[0]
    histories_one_step = histories_dict[node][0]
    x_min = scene.x_min + histories_one_step[0] - 100.0
    y_min = scene.y_min + histories_one_step[1] - 50.0
    x_max = scene.x_min + histories_one_step[0] + 100.0
    y_max = scene.y_min + histories_one_step[1] + 100.0

    # Plot predicted timestep for random scene in map
    my_patch = (x_min, y_min, x_max, y_max)
    layers = ['drivable_area',
              'road_segment',
              'lane',
              'ped_crossing',
              'walkway',
              'stop_line',
              'road_divider',
              'lane_divider']

    fig, ax = nusc_map.render_map_patch(my_patch, layers, figsize=(23, 15), alpha=0.1, render_egoposes_range=False)

    # Plot predicted timestep
    plotting_helper.plot_vehicle_dist(ax,
                                      pred_dists,
                                      scene,
                                      max_hl=max_hl,
                                      ph=ph,
                                      x_min=scene.x_min,
                                      y_min=scene.y_min,
                                      line_width=0.5,
                                      car_img_zoom=0.02,
                                      robot_plan=q_star)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    # plt.show()
    fig.savefig('plots/scene_'+str(scene_num)+'_t_'+str(first_ts)+'.png', dpi=300, bbox_inches='tight')
