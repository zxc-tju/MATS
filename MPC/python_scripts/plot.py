import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
from scipy.ndimage import rotate
from scipy import linalg
import seaborn as sns
import torch
import dill
import json

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap

sys.path.append("../../mats")
from model.model_registrar import ModelRegistrar
from model.mats import MATS


line_colors = ['#375397',
#                '#F05F78',
               '#80CBE5',
               '#ABCB51',
               '#C8B0B0']

cars = [plt.imread('icons/Car TOP_VIEW 375397.png'),
#         plt.imread('icons/Car TOP_VIEW F05F78.png'),
        plt.imread('icons/Car TOP_VIEW 80CBE5.png'),
        plt.imread('icons/Car TOP_VIEW ABCB51.png'),
        plt.imread('icons/Car TOP_VIEW C8B0B0.png')]

robot = plt.imread('icons/Car TOP_VIEW ROBOT.png')


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    mats = MATS(model_registrar, hyperparams, None, 'cpu')
    return mats, hyperparams


def load_data():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    nuScenes_data_path = '/experiments/nuScenes/data'
    nusc = NuScenes(version='v1.0-trainval', dataroot=nuScenes_data_path, verbose=True)
    helper = PredictHelper(nusc)

    # ns_scene = nusc.get('scene', nusc.field2token('scene', 'name', 'scene-0024')[0])

    layers = ['drivable_area',
              'road_segment',
              'lane',
              'ped_crossing',
              'walkway',
              'stop_line',
              'road_divider',
              'lane_divider']

    with open('../../experiments/processed/nuScenes_val_full.pkl', 'rb') as f:
        env = dill.load(f, encoding='latin1')

    # Modeling Loading
    mats, hyperparams = load_model(
        '../../experiments/nuScenes/models/models_26_Jul_2020_17_11_34_full_zeroRrows_batch8_fixed_edges',
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

    return scenes


def prediction_output_to_trajectories(prediction_output_dict,
                                      max_h,
                                      ph,
                                      map=None,
                                      prune_ph_to_future=False):

    prediction_timesteps = prediction_output_dict.keys()

    output_dict = dict()
    histories_dict = dict()
    futures_dict = dict()

    for t in prediction_timesteps:
        histories_dict[t] = dict()
        output_dict[t] = dict()
        futures_dict[t] = dict()
        prediction_nodes = prediction_output_dict[t].keys()
        for node in prediction_nodes:
            predictions_output = prediction_output_dict[t][node]
            position_state = {'position': ['x', 'y']}

            history = node.get(np.array([t - max_h, t]), position_state)  # History includes current pos
            history = history[~np.isnan(history.sum(axis=1))]

            future = node.get(np.array([t + 1, t + ph]), position_state)
            future = future[~np.isnan(future.sum(axis=1))]

            if prune_ph_to_future:
                predictions_output = predictions_output[:, :, :future.shape[0]]
                if predictions_output.shape[2] == 0:
                    continue

            trajectory = predictions_output

            if map is None:
                histories_dict[t][node] = history
                output_dict[t][node] = trajectory
                futures_dict[t][node] = future
            else:
                histories_dict[t][node] = map.to_map_points(history)
                output_dict[t][node] = trajectory, map.to_map_points(trajectory.component_distribution.mean)
                futures_dict[t][node] = map.to_map_points(future)

    return output_dict, histories_dict, futures_dict


def plot_vehicle_nice(ax, predictions, scene,
                      max_hl=10, ph=6,
                      map=None, x_min=0, y_min=0,
                      pi_alpha=False, line_alpha=0.8,
                      line_width=0.2, edge_width=2,
                      circle_edge_width=0.5, node_circle_size=0.3,
                      car_img_zoom=0.01,
                      plot_prediction=True, plot_future=True,
                      at_timestep=0):
    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(predictions,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)
    assert (len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.fdata, origin='lower', alpha=0.5)

    cmap = ['k', 'b', 'y', 'g', 'r']
    a = []
    i = 0
    node_list = sorted(histories_dict.keys(), key=lambda x: x.id)
    for node in node_list:
        history = histories_dict[node] + np.array([x_min, y_min])
        future = futures_dict[node] + np.array([x_min, y_min])
        predictions = prediction_dict[node]
        if node.type.name == 'VEHICLE':
            # ax.plot(history[:, 0], history[:, 1], 'ko-', linewidth=1)

            if plot_future:
                ax.plot(future[at_timestep:, 0],
                        future[at_timestep:, 1],
                        'w--o',
                        linewidth=4,
                        markersize=3,
                        zorder=650,
                        path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])

            if plot_prediction:
                pis = predictions.pis
                for component in range(predictions.mixture_distribution.param_shape[-1]):
                    if pi_alpha:
                        line_alpha = pis[component].item()

                    ax.plot(predictions.component_distribution.mean[at_timestep:, 0, component, 0] + x_min,
                            predictions.component_distribution.mean[at_timestep:, 0, component, 1] + y_min,
                            color=cmap[1],
                            linewidth=line_width,
                            alpha=line_alpha,
                            zorder=600)

            vel = node.get(np.array([ts_key + at_timestep]), {'velocity': ['x', 'y']})
            h = np.arctan2(vel[0, 1], vel[0, 0])
            r_img = rotate(cars[i % len(cars)],
                           node.get(np.array([ts_key + at_timestep]), {'heading': ['°']})[0, 0] * 180 / np.pi,
                           reshape=True)
            oi = OffsetImage(r_img, zoom=car_img_zoom, zorder=700)
            if at_timestep > 0:
                veh_box = AnnotationBbox(oi, (future[at_timestep - 1, 0], future[at_timestep - 1, 1]), frameon=False)
            else:
                veh_box = AnnotationBbox(oi, (history[-1, 0], history[-1, 1]), frameon=False)

            veh_box.zorder = 700
            ax.add_artist(veh_box)
            i += 1
        else:
            # ax.plot(history[:, 0], history[:, 1], 'k--')

            if plot_prediction:
                pis = predictions.pis
                for component in range(predictions.mixture_distribution.param_shape[-1]):
                    if pi_alpha:
                        line_alpha = pis[component].item()

                    ax.plot(predictions.component_distribution.mean[at_timestep:, 0, component, 0] + x_min,
                            predictions.component_distribution.mean[at_timestep:, 0, component, 1] + y_min,
                            color=cmap[1],
                            linewidth=line_width,
                            alpha=line_alpha,
                            zorder=600)

            if plot_future:
                ax.plot(future[at_timestep:, 0],
                        future[at_timestep:, 1],
                        'w--',
                        zorder=650,
                        path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            # Current Node Position
            if at_timestep > 0:
                x, y = future[at_timestep - 1, 0], future[at_timestep - 1, 1]
            else:
                x, y = history[-1, 0], history[-1, 1]
            circle = plt.Circle((x, y),
                                node_circle_size,
                                facecolor='g',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

    # Visualizing the ego-vehicle as well.
    position_state = {'position': ['x', 'y']}
    history = scene.robot.get(np.array([ts_key - max_hl, ts_key]), position_state)  # History includes current pos
    history = history[~np.isnan(history.sum(axis=1))]
    history += np.array([x_min, y_min])

    future = scene.robot.get(np.array([ts_key + 1, ts_key + ph]), position_state)
    future = future[~np.isnan(future.sum(axis=1))]
    future += np.array([x_min, y_min])

    if plot_future:
        ax.plot(future[at_timestep:, 0],
                future[at_timestep:, 1],
                'w--o',
                linewidth=4,
                markersize=3,
                zorder=650,
                path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])

    vel = scene.robot.get(np.array([ts_key + at_timestep]), {'velocity': ['x', 'y']})
    h = np.arctan2(vel[0, 1], vel[0, 0])
    r_img = rotate(robot, scene.robot.get(np.array([ts_key + at_timestep]), {'heading': ['°']})[0, 0] * 180 / np.pi,
                   reshape=True)
    oi = OffsetImage(r_img, zoom=car_img_zoom, zorder=700)

    if at_timestep > 0:
        veh_box = AnnotationBbox(oi, (future[at_timestep - 1, 0], future[at_timestep - 1, 1]), frameon=False)
    else:
        veh_box = AnnotationBbox(oi, (history[-1, 0], history[-1, 1]), frameon=False)
    veh_box.zorder = 700
    ax.add_artist(veh_box)