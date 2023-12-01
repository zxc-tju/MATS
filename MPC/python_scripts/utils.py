import sys
import os
import dill
import glob
import json
import random
import torch
import numpy as np
import pandas as pd

sys.path.append("../../mats")
from model.model_registrar import ModelRegistrar
from model.mats import MATS
from environment.node import MultiNode
from model.dataset import EnvironmentDataset
import evaluation
import visualization

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# def load_model(model_dir, env, ts=100):
#     model_registrar = ModelRegistrar(model_dir, 'cpu')
#     model_registrar.load_models(ts)
#     with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
#         hyperparams = json.load(config_json)
#     mats = MATS(model_registrar, hyperparams, None, 'cpu')
#     return mats, hyperparams

def load_model(model_path, env, ts=100):
    model_registrar = ModelRegistrar(model_path, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_path, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)
    mats = MATS(model_registrar, hyperparams, None, 'cpu')

    # Set the environment and annealing parameters
    mats.set_environment(env)
    mats.set_annealing_params()

    # Calculate scene graphs
    calculate_scene_graphs(env, hyperparams)

    return mats, hyperparams


def load_data_set(filename):
    with open(filename, 'rb') as f:
        env = dill.load(f, encoding='latin1')
    return env


def calculate_scene_graphs(env, hyperparams):
    scenes = env.scenes
    for scene in scenes:
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])


def predict(mats, hyperparams, scene, timestep, num_modes):
    ph = hyperparams['prediction_horizon']

    with torch.no_grad():
        mats_outputs = mats.predict(
            scene,
            np.array([timestep]),
            ph,
            min_future_timesteps=ph,
            include_B=hyperparams['include_B'],
            zero_R_rows=hyperparams['zero_R_rows'])

    pred_dists, non_rob_rows, As, Bs, Qs, affine_terms, state_lengths_in_order = mats_outputs

    A = As[0].numpy()
    B = Bs[0].numpy()
    Q = Qs[0].numpy()
    affine_terms = affine_terms[0].numpy()

    prediction_info = dict()

    state_lengths_in_order = state_lengths_in_order.squeeze().numpy()
    current_state_idx = state_lengths_in_order[0]
    for idx, node in enumerate(pred_dists[timestep]):
        curr_state = node.get(np.array([timestep, timestep]), hyperparams['pred_state'][node.type.name])
        node_predictions = pred_dists[timestep][node]
        state_predictions = node_predictions.component_distribution.mean.numpy()
        mode_probs = node_predictions.pis.numpy()
        rank_order = np.argsort(mode_probs)[::-1]
        node_str = '/'.join([node.type.name, str(node.id)])
        prediction_info[node_str] = {'node_type': node.type.name,
                                     'node_idx': idx + 1,
                                     'current_state': curr_state,
                                     'mode_probs': mode_probs[rank_order[:num_modes]],
                                     'state_predictions': state_predictions[:, 0, rank_order[:num_modes]],
                                     'state_uncertainties': Q[:, 0, rank_order[:num_modes],
                                                            current_state_idx:current_state_idx +
                                                                              state_lengths_in_order[idx + 1]]}
        current_state_idx += state_lengths_in_order[idx + 1]

    dynamics_dict = {'A': A[:, 0, rank_order[:num_modes]],
                     'B': B[:, 0, rank_order[:num_modes]],
                     'affine_terms': affine_terms[:, 0, rank_order[:num_modes]]}

    return prediction_info, dynamics_dict, mats_outputs


def predicted_dynamics(pred_settings, robot_current_state, scene_num, timestep):
    # Call the predict function from python_utils module
    prediction_info, dynamics_dict, mats_outputs = predict(pred_settings.mats,
                                                           pred_settings.hyperparams,
                                                           pred_settings.env.scenes[scene_num],
                                                           timestep - 1,
                                                           pred_settings.num_modes)

    # Retrieve the prediction horizon and number of modes from settings
    pred_horizon = pred_settings.hyperparams.get("prediction_horizon")
    num_modes = pred_settings.num_modes

    # Extract dynamics matrices A, B, and affine terms g
    Aps = [[dynamics_dict["A"][ts, mode, :, :] for ts in range(pred_horizon)] for mode in range(num_modes)]
    Bps = [[dynamics_dict["B"][ts, mode, :, :] for ts in range(pred_horizon)] for mode in range(num_modes)]
    gps = [[dynamics_dict["affine_terms"][ts, mode, :] for ts in range(pred_horizon)] for mode in range(num_modes)]

    # Define state dimension (TODO: this should be retrieved from settings or environment)
    state_dim = 4

    # Initialize state vector q0
    q0 = np.zeros((len(prediction_info) + 1) * state_dim)
    ordered_node_ids = [None] * len(prediction_info)

    # Populate q0 and ordered_node_ids based on prediction_info
    for k, v in prediction_info.items():
        node_idx = v["node_idx"]
        state_vector_idx = node_idx * state_dim
        q0[state_vector_idx:state_vector_idx + state_dim] = v["current_state"]
        ordered_node_ids[node_idx - 1] = k.split("/")[1]

    # Populate robot state in q0
    # robot = pred_settings.env.scenes[scene_num].robot
    # q_robot = [robot.data.data[timestep, 0],  # x
    #            robot.data.data[timestep, 1],  # y
    #            robot.data.data[timestep, 8],  # heading
    #            robot.data.data[timestep, 10]]  # v
    q0[:state_dim] = robot_current_state

    return Aps, Bps, gps, q0, ordered_node_ids, mats_outputs



# TODO: hard coded indices
# TODO: check direction of accel vector so there's nothing weird happening with accel norm
def get_recorded_robot_controls(pred_settings, scene_num, timestep):
    omega = pred_settings.env.scenes[scene_num].robot.data.data[timestep:, 9]
    a = pred_settings.env.scenes[scene_num].robot.data.data[timestep:, 11]
    return np.column_stack((omega, a))


# TODO: hardcoded indices
def predict_future_states(pred_settings, q0, us, As, Bs, gs, mode):
    pred_horizon = pred_settings.hyperparams.get("prediction_horizon")
    state_dim = As[0][0].shape[0]
    q_pred = np.zeros((state_dim, pred_horizon + 1))
    q_pred[:, 0] = q0

    for k in range(pred_horizon):
        q_pred[:, k + 1] = gs[mode][k] + np.dot(As[mode][k], q_pred[:, k]) + np.dot(Bs[mode][k], us[k, :])

    return q_pred


# TODO: hardcoded indices
def update_obstacles_from_predictions(q_pred, present_node_ids, vals, scene, iter=None):
    obstacle_heading = {obs_id: [] for obs_id in vals.obstacles.keys()}

    for mode in range(vals.num_modes):
        for history_node_id, obs in vals.obstacles.items():
            if history_node_id in present_node_ids:
                node_idx = present_node_ids.index(history_node_id) + 1  # very important: +1 to avoid robot node!
                row_idx = node_idx * 4
                vals.obstacles[history_node_id].positions[mode] = [
                    [q_pred[mode][row_idx, k] + scene.x_min, q_pred[mode][row_idx + 1, k] + scene.y_min] for k in
                    range(vals.obstacle_horizon)]
                vals.obstacles[history_node_id].active = True
                obstacle_heading[history_node_id].append(q_pred[mode][row_idx + 2, :])
            else:
                vals.obstacles[history_node_id].active = False

    # # Save obstacle heading data
    # filename = f"data/obstacle_heading_iteration_{1 if iter is None else iter}"
    # np.save(filename, obstacle_heading)


def end_horizon_idces(vals):
    n_modes = vals.num_modes
    N = vals.horizon
    k_c = vals.consensus_horizon
    return [N + (N - k_c) * (i - 1) for i in range(1, n_modes + 1)]
