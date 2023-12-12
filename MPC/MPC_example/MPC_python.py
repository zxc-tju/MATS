import sys
import os
import time
import xlsxwriter
import pandas as pd
import numpy as np
from datetime import datetime

from tqdm import tqdm, trange
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap

sys.path.append("../../mats")
sys.path.append("../../")
from MPC.python_scripts.utils import (load_model, load_data_set, predicted_dynamics, get_recorded_robot_controls,
                                      update_obstacles_from_predictions, predict_future_states)
from MPC.python_scripts.structs import (get_scene_info, ControlLimits, DynamicsModel,
                                        PredictionSettings, init_node_obstacles)
from MPC.python_scripts.path_handling import load_splines, SplinePath, find_best_s, get_path_obj
from MPC.python_scripts.mpc import MPCProblem, MPCValues, initial_guess
import MPC.python_scripts.plot as plotting_helper

from utils import prediction_output_to_trajectories

" ==== Prepare model and data ==== "
# Load processed data
env = load_data_set("/home/zxc/codes/MATS/experiments/processed/nuScenes_train_val_full_doubled.pkl")
# env = load_data_set("/home/zxc/codes/MATS/experiments/processed/nuScenes_val_full_doubled.pkl")

# Raw data for plots
# nuScenes_data_path = '/home/zxc/codes/MATS/experiments/nuScenes/data'  # for home
nuScenes_data_path = '/home/zxc/Downloads/nuscene'  # for 423

# load model
mats, hyperparams = load_model('../../experiments/nuScenes/models'
                               '/models_26_Jul_2020_17_11_34_full_zeroRrows_batch8_fixed_edges',
                               env, ts=16)
# mats, hyperparams = load_model('../../experiments/nuScenes/models'
#                                '/models_21_Jul_2020_10_25_10_full_zeroRrows_batch8_double_fixed_a_norm',
#                                env, ts=11)
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

# Prepare nuscenes map for selected scene
nusc = NuScenes(version='v1.0-trainval', dataroot=nuScenes_data_path, verbose=True)
helper = PredictHelper(nusc)


def run_mpc(scene_num=None, planner=None, save_meta=False, plot_fig=False,
            t_range=range(2, 20), my_patch=None,
            plot_path=None):
    " ==== Prepare Data Recording and Plotting ==== "
    # Process data
    mats_outputs_collection = []
    robot_state_collection = []
    robot_plan_collection = []
    solving_time_consumption_collection = []
    full_planning_time_consumption_collection = []

    # t_range = range(2, 20)
    if not os.path.isdir('./data'):
        os.mkdir('./data')

    if not os.path.isdir('./plots'):
        os.mkdir('./plots')

    layers = ['drivable_area',
              'road_segment',
              'lane',
              'ped_crossing',
              'walkway',
              'stop_line',
              'road_divider',
              'lane_divider']

    " ==== Select Scene ==== "
    # select scene
    # scene_num = scene_num  # original demo is 23
    scene = scenes[scene_num]
    scene.calculate_scene_graph(env.attention_radius,
                                hyperparams['edge_addition_filter'],
                                hyperparams['edge_removal_filter'])

    # Prepare nuscenes map for selected scene
    nusc_map = NuScenesMap(dataroot=nuScenes_data_path, map_name=helper.get_map_name_from_sample_token(scene.name))

    robot_node, non_robot_nodes, non_robot_node_ids = get_scene_info(env, scene_num)

    # Configure figure center
    if my_patch is None:
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
    iteration_num = 5

    " ==== MPC process ==== "

    " --- Initialize MPC --- "
    first_ts = t_range[0]
    state_star = []
    # robot initial state
    q0 = [robot_node.x[first_ts], robot_node.y[first_ts], robot_node.theta[first_ts], robot_node.v[first_ts], 0]
    q0[4] = find_best_s(q0, path_obj, enable_global_search=True)
    robot_state_collection.append(q0)

    mpc_vals_obj = MPCValues(path_obj, initial_state=q0,
                             num_modes=num_modes, horizon=7,
                             consensus_horizon=4, num_obstacles=len(non_robot_nodes), timestep=scene.dt)

    # make first predictions for obstacle constraints
    init_node_obstacles(non_robot_node_ids, mpc_vals_obj)
    Aps, Bps, gps, q_pred0, nodes_present, mats_outputs = predicted_dynamics(pred_settings, q0[:4], scene_num, first_ts)
    u_pred = get_recorded_robot_controls(pred_settings, scene_num, first_ts)
    q_pred = [predict_future_states(pred_settings, q_pred0, u_pred, Aps, Bps, gps, j) for j in
              range(0, mpc_vals_obj.num_modes)]

    update_obstacles_from_predictions(q_pred, nodes_present, mpc_vals_obj, scene)

    # # save prediction results
    # mats_outputs_collection.append(mats_outputs)

    # initial solution guess
    initial_state_plan, initial_control_plan = initial_guess(mpc_vals_obj)

    " --- Solve MPC --- "
    for ts in tqdm(t_range[1:]):
        time_start = time.time()
        for i in range(iteration_num):
            # construct MPC
            mpc = MPCProblem(dynamics_obj, mpc_vals_obj, non_robot_node_ids,
                             initial_state_plan, initial_control_plan, planner=planner)

            # solve problem
            time_start_solving = time.time()
            state_star, control_star = mpc.solve()  # state_star include current state
            # print(f'=====Results of {i} Iteration=====')
            # print('output states', state_star[-1, 0:3])
            time_solved = time.time()
            solving_time_consumption_collection.append(time_solved - time_start_solving)

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

        # update prediction on system dynamics in the new step ts
        Aps, Bps, gps, q_pred0, nodes_present, mats_outputs = predicted_dynamics(pred_settings,
                                                                                 initial_state_plan[:4, 1],
                                                                                 scene_num, ts)

        # prediction on obstacles (without map offset)
        u_pred = initial_control_plan[0:2, 0:prediction_horizon]  # use control from optimization solution
        u_pred = u_pred.T
        q_pred = [predict_future_states(pred_settings, q_pred0, u_pred, Aps, Bps, gps, j) for j in
                  range(0, mpc_vals_obj.num_modes)]
        # align obstacles' positions with map offset
        update_obstacles_from_predictions(q_pred, nodes_present, mpc_vals_obj, scene)

        # shift [states, controls, dynamics] to the next time step
        mpc_vals_obj.update_problem(initial_state_plan, initial_control_plan)

        time_finish = time.time()

        # save planning time
        full_planning_time_consumption_collection.append(time_finish - time_start)

        # save plan
        robot_plan_collection.append(state_star.copy())
        robot_state_collection.append(state_star.copy()[:, 1])

        # save prediction results
        mats_outputs_collection.append(mats_outputs)

    " ==== Visualize Results ==== "
    if plot_fig:
        # Plot predicted timestep
        plotting_helper.plot_multi_frame_dist(my_patch, layers, mats_outputs_collection, scene, nusc_map=nusc_map,
                                              max_hl=max_hl, ph=prediction_horizon, x_min=scene.x_min,
                                              y_min=scene.y_min,
                                              robot_plan=robot_plan_collection, scene_num=scene_num,
                                              plot_path=plot_path)
        plotting_helper.plot_A(my_patch, layers, mats_outputs_collection, scene, nusc_map=nusc_map, max_hl=max_hl,
                               ph=prediction_horizon, x_min=scene.x_min, y_min=scene.y_min,
                               robot_plan=robot_plan_collection, scene_num=scene_num,
                               plot_path=plot_path)
        plotting_helper.plot_B(my_patch, layers, mats_outputs_collection, scene, nusc_map=nusc_map, max_hl=max_hl,
                               ph=prediction_horizon, x_min=scene.x_min, y_min=scene.y_min,
                               robot_plan=robot_plan_collection, scene_num=scene_num,
                               plot_path=plot_path)

    " ==== Analyze and Save Meta ==== "
    if save_meta:
        # we have recorded data of robot_plan_collection, robot_state_collection, mats_outputs_collection
        # Settings
        position_state = {'position': ['x', 'y']}

        # refresh for each scene
        min_distance_in_event = 999

        robot_state_collection_array = np.array(robot_state_collection)
        robot_trajectory = robot_state_collection_array[:, 0:2]
        for i in range(len(mats_outputs_collection)):
            min_distance_in_plan = 999
            scenario_time_list = list(mats_outputs_collection[i][0].keys())
            scenario_time = scenario_time_list[0]

            # robot info
            robot_plan_array = np.array(robot_plan_collection[i]).T
            robot_plan = robot_plan_array[:prediction_horizon + 1, 0:2]
            progress_in_plan = robot_plan_array[prediction_horizon, 4] - robot_plan_array[0, 4]

            node_id_list = mats_outputs_collection[i][0][scenario_time].keys()
            for node in node_id_list:

                node_future_trajectory = (
                        node.get(np.array([scenario_time, scenario_time + prediction_horizon]), position_state)
                        + np.array([scene.x_min, scene.y_min]))

                # distance from ego plan to others' GT future
                distance_in_plan = np.linalg.norm(robot_plan - node_future_trajectory, axis=1)
                if np.min(distance_in_plan) < min_distance_in_plan:
                    min_distance_in_plan = np.min(distance_in_plan)

                # distance between current positions
                actual_dis = distance_in_plan[0]
                if actual_dis < min_distance_in_event:
                    min_distance_in_event = actual_dis

            min_distance_in_plans_collection.append(min_distance_in_plan)
            progress_in_plans_collection.append(progress_in_plan)

        min_distance_in_event_collection.append(min_distance_in_event)
        average_solving_time_collection.append(np.mean(np.array(solving_time_consumption_collection)))
        average_planning_time_collection.append(np.mean(np.array(full_planning_time_consumption_collection)))
        progress_in_event_collection.append(robot_state_collection_array[-1, 4])
        final_state_ground_truth = [robot_node.x[t_range[-1]], robot_node.y[t_range[-1]],
                                    robot_node.theta[t_range[-1]], robot_node.v[t_range[-1]], 0]
        progress_ground_truth.append(find_best_s(final_state_ground_truth, path_obj, enable_global_search=True))

    print(f' ===== Scene {scene_num} Finished ===== ')


if __name__ == '__main__':

    # Event-level meta
    min_distance_in_event_collection = []
    progress_in_event_collection = []
    progress_ground_truth = []
    average_solving_time_collection = []
    average_planning_time_collection = []
    successful_scene_id_list = []

    # Frame-level meta
    min_distance_in_plans_collection = []
    progress_in_plans_collection = []

    " ==== Single-Scene Running ==== "
    # train_val_full_doubled 16: many pedestrian t=(2,24)  ROI=[870, 1590, 930, 1640]
    # val_full_doubled 23: cross interaction t=(2,18)  ROI=None

    # scene_id = 23
    # run_mpc(scene_num=scene_id, plot_fig=True, t_range=range(2, 18),
    #         my_patch=None,
    #         plot_path='/home/zxc/codes/MATS/MPC/MPC_example/plots')

    scene_id = {7, 16, 18, 30, 39, 57, 61, 62, 73, 74, 85, 92, 96}
    for idx in scene_id:
        run_mpc(scene_num=idx, plot_fig=True,
                plot_path='/home/zxc/codes/MATS/MPC/MPC_example/plots')

    " ==== Multi-Scene Running ==== "
    # # planner_type = 'MPC-5-Iteration'  # change the iteration number accordingly
    # # planner_type = 'MPC'  # change the iteration number accordingly
    # planner_type = 'Single-Obstacle'  # this type will change obstacle constraints in mpc.py  Iter=5
    # print('Start' + planner_type)
    #
    # # Get the current date and time
    # current_datetime = datetime.now()
    # formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
    # print("Date and Time:", formatted_datetime)
    #
    # workbook_path = ('/home/zxc/codes/MATS/MPC/MPC_example/data/' + planner_type + '-' + formatted_datetime + '.xlsx')
    # workbook = xlsxwriter.Workbook(workbook_path)
    # worksheet = workbook.add_worksheet()
    # workbook.close()
    #
    # scene_id_list = range(185)
    # for scene_id in scene_id_list:
    #     try:
    #         run_mpc(scene_num=scene_id, planner=planner_type, save_meta=True)
    #         successful_scene_id_list.append(scene_id)
    #     except:
    #         print(f'+++++-----Scene {scene_id} Failed-----+++++')
    #         continue
    #
    # # prepare data_event
    # data_event = {'scene_id': successful_scene_id_list,
    #               'min_distance_in_event_collection': min_distance_in_event_collection,
    #               'progress_in_event_collection': progress_in_event_collection,
    #               'progress_ground_truth': progress_ground_truth,
    #               'average_solving_time_collection': average_solving_time_collection,
    #               'average_planning_time_collection': average_planning_time_collection}
    #
    # pd_event = pd.DataFrame(data_event)
    #
    # # prepare data_frame
    # data_frame = {'min_distance_in_plans_collection': min_distance_in_plans_collection,
    #               'progress_in_plans_collection': progress_in_plans_collection}
    # pd_frame = pd.DataFrame(data_frame)
    #
    # # write data_event
    # with pd.ExcelWriter(workbook_path, mode='a', if_sheet_exists="overlay", engine="openpyxl") as writer:
    #
    #     pd_event.to_excel(writer, index=False, header=True, sheet_name='event', startcol=0,
    #                       startrow=0)
    #     pd_frame.to_excel(writer, index=False, header=True, sheet_name='frame', startcol=0,
    #                       startrow=0)
    #
    #     writer.close()
