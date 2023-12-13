import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.transforms as transforms
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
from scipy.ndimage import rotate
from scipy import linalg
from tqdm import tqdm, trange
import seaborn as sns
import torch
import dill
import json

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap

sys.path.append("../../mats")
# from model.model_registrar import ModelRegistrar
from utils import calculate_A_slices, calculate_BQ_slices

# # from model.mats import MATS
LINE_ALPHA = 0.8
LINE_WIDTH = 0.2
EDGE_WIDTH = 2
CIRCLE_EDGE_WIDTH = 0
NODE_CIRCLE_SIZE = 0.4
CAR_IMG_ZOOM = 0.02
VEHICLE_WIDTH, VEHICLE_HEIGHT = 2, 4
FUTURE_LINE_WIDTH = 5

line_colors = ['#375397',
               #                '#F05F78',
               '#80CBE5',
               '#ABCB51',
               '#C8B0B0']

cars = [plt.imread('../images/icons/Car TOP_VIEW 375397.png'),
        #         plt.imread('../images/icons/Car TOP_VIEW F05F78.png'),
        plt.imread('../images/icons/Car TOP_VIEW 80CBE5.png'),
        plt.imread('../images/icons/Car TOP_VIEW ABCB51.png'),
        plt.imread('../images/icons/Car TOP_VIEW C8B0B0.png')]

robot = plt.imread('../images/icons/Car TOP_VIEW ROBOT.png')

cmap = ['k', 'b', 'y', 'g', 'r']


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


def plot_vehicle_dist(ax, predictions, scene,
                      max_hl=10, ph=6,
                      map=None, x_min=0, y_min=0,
                      pi_alpha=False, line_alpha=0.8,
                      line_width=0.2, edge_width=2,
                      circle_edge_width=0.5, node_circle_size=0.3,
                      car_img_zoom=0.01, pi_threshold=0.05,
                      robot_plan=None):
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
        pis = predictions.pis

        if node.type.name == 'VEHICLE':
            ax.plot(future[:, 0],
                    future[:, 1],
                    'w--o',
                    linewidth=4,
                    markersize=3,
                    zorder=650,
                    path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])

            for component in range(predictions.mixture_distribution.param_shape[-1]):
                pi = pis[component].item()
                if pi < pi_threshold:
                    continue

                means = predictions.component_distribution.mean[:, 0, component, :2] + np.array([x_min, y_min])
                covs = predictions.component_distribution.covariance_matrix[:, 0, component, :2, :2]

                ax.plot(means[..., 0], means[..., 1],
                        '-o', markersize=3,
                        color=cmap[1],
                        linewidth=line_width,
                        alpha=line_alpha,
                        zorder=620)

                for timestep in range(means.shape[0]):
                    mean = means[timestep]
                    covar = covs[timestep]

                    v, w = linalg.eigh(covar)
                    v = 2. * np.sqrt(2.) * np.sqrt(v)
                    u = w[0] / linalg.norm(w[0])

                    # Plot an ellipse to show the Gaussian component
                    angle = np.arctan2(u[1], u[0])
                    angle = 180. * angle / np.pi  # convert to degrees
                    ell = patches.Ellipse(mean, v[0], v[1], 180. + angle, color='blue', zorder=600)
                    ell.set_edgecolor(None)
                    ell.set_clip_box(ax.bbox)
                    ell.set_alpha(pi / 3.)
                    ax.add_artist(ell)

            vel = node.get(np.array([ts_key]), {'velocity': ['x', 'y']})
            h = np.arctan2(vel[0, 1], vel[0, 0])
            r_img = rotate(cars[i % len(cars)], node.get(np.array([ts_key]), {'heading': ['°']})[0, 0] * 180 / np.pi,
                           reshape=True)
            oi = OffsetImage(r_img, zoom=car_img_zoom, zorder=700)
            veh_box = AnnotationBbox(oi, (history[-1, 0], history[-1, 1]), frameon=False)
            veh_box.zorder = 700
            ax.add_artist(veh_box)
            i += 1
        else:
            for component in range(predictions.mixture_distribution.param_shape[-1]):
                pi = pis[component].item()
                if pi < pi_threshold:
                    continue

                means = predictions.component_distribution.mean[:, 0, component, :2] + np.array([x_min, y_min])
                covs = predictions.component_distribution.covariance_matrix[:, 0, component, :2, :2]

                ax.plot(means[..., 0], means[..., 1],
                        '-o', markersize=3,
                        color=cmap[1],
                        linewidth=line_width,
                        alpha=line_alpha,
                        zorder=620)

                for timestep in range(means.shape[0]):
                    mean = means[timestep]
                    covar = covs[timestep]

                    v, w = linalg.eigh(covar)
                    v = 2. * np.sqrt(2.) * np.sqrt(v)
                    u = w[0] / linalg.norm(w[0])

                    # Plot an ellipse to show the Gaussian component
                    angle = np.arctan2(u[1], u[0])
                    angle = 180. * angle / np.pi  # convert to degrees
                    ell = patches.Ellipse(mean, v[0], v[1], 180. + angle, color='blue', zorder=600)
                    ell.set_edgecolor(None)
                    ell.set_clip_box(ax.bbox)
                    ell.set_alpha(pi / 3.)
                    ax.add_artist(ell)

            ax.plot(future[:, 0],
                    future[:, 1],
                    'w--',
                    zorder=650,
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            # Current Node Position
            circle = plt.Circle((history[-1, 0],
                                 history[-1, 1]),
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

    ax.plot(future[:, 0],
            future[:, 1],
            'r-o',
            linewidth=4,
            markersize=3,
            zorder=650,
            path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])

    # if robot_plan is not None:
    #     future_plan = robot_plan[0:2, :6]
    #     future_plan = future_plan.T
    #     ax.plot(future_plan[:, 0],
    #             future_plan[:, 1],
    #             'r--o',
    #             linewidth=4,
    #             markersize=3,
    #             zorder=650,
    #             path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])

    vel = scene.robot.get(np.array([ts_key]), {'velocity': ['x', 'y']})
    h = np.arctan2(vel[0, 1], vel[0, 0])
    r_img = rotate(robot, scene.robot.get(np.array([ts_key]), {'heading': ['°']})[0, 0] * 180 / np.pi,
                   reshape=True)
    oi = OffsetImage(r_img, zoom=car_img_zoom, zorder=700)
    veh_box = AnnotationBbox(oi, (history[-1, 0], history[-1, 1]), frameon=False)
    veh_box.zorder = 700
    ax.add_artist(veh_box)


def plot_multi_frame_dist(patch, layers, mats_outputs_collection, scene,
                          nusc_map=None, max_hl=10, ph=6, pi_threshold=0.05,
                          x_min=0, y_min=0,
                          robot_plan=None, scene_num=None,
                          plot_path=None):
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    plot_path = plot_path + '/scene_' + str(scene_num) + '/'
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    for t in range(len(mats_outputs_collection)):
        fig, ax = nusc_map.render_map_patch(patch, layers, figsize=(23, 15), alpha=0.06, render_egoposes_range=False)

        # get prediction results
        prediction_distributions = mats_outputs_collection[t][0]
        prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_distributions,
                                                                                          max_hl,
                                                                                          ph,
                                                                                          map=None)

        assert (len(prediction_dict.keys()) <= 1)
        if len(prediction_dict.keys()) == 0:
            print("t", t)
        ts_key = list(prediction_dict.keys())[0]

        prediction_dict = prediction_dict[ts_key]
        histories_dict = histories_dict[ts_key]
        futures_dict = futures_dict[ts_key]

        node_list = sorted(histories_dict.keys(), key=lambda x: x.id)
        for node in node_list:
            history = histories_dict[node] + np.array([x_min, y_min])
            future = futures_dict[node] + np.array([x_min, y_min])
            predictions = prediction_dict[node]
            pis = predictions.pis  # the possibility of a mode

            if node.type.name == 'VEHICLE':

                # Gaussian Distribution
                for mode_idx in range(predictions.mixture_distribution.param_shape[-1]):
                    possibility = pis[mode_idx].item()
                    if possibility < pi_threshold:
                        continue

                    means = predictions.component_distribution.mean[:, 0, mode_idx, :2] + np.array([x_min, y_min])
                    covariances = predictions.component_distribution.covariance_matrix[:, 0, mode_idx, :2, :2]

                    plot_node_Gaussian_distribution(means, covariances, ax, possibility)

                # Future
                # Create a custom color gradient from dark blue to light yellow
                colors = [plt.cm.viridis(i / len(future)) for i in range(len(future))]

                for i in range(np.size(future, 0) - 1):
                    # Draw the segment
                    ax.plot([future[i, 0], future[i + 1, 0]],
                            [future[i, 1], future[i + 1, 1]],
                            color=colors[i], linewidth=FUTURE_LINE_WIDTH / 2, zorder=650)

                # Current position
                angle = node.get(np.array([ts_key]), {'heading': ['°']})[0, 0] * 180 / np.pi - 90
                plot_rectangle(current_position=history[-1, :], angle=angle, ax=ax, color='gray', alpha=0.7)

            elif node.type.name in {'PEDESTRIAN'}:

                # Gaussian Distribution
                for mode_idx in range(predictions.mixture_distribution.param_shape[-1]):
                    possibility = pis[mode_idx].item()
                    if possibility < pi_threshold:
                        continue

                    means = predictions.component_distribution.mean[:, 0, mode_idx, :2] + np.array([x_min, y_min])
                    covariances = predictions.component_distribution.covariance_matrix[:, 0, mode_idx, :2, :2]

                    plot_node_Gaussian_distribution(means, covariances, ax, possibility)

                # Future
                # Create a custom color gradient from dark blue to light yellow
                colors = [plt.cm.viridis(i / len(future)) for i in range(len(future))]

                for i in range(np.size(future, 0) - 1):
                    # Draw the segment
                    ax.plot([future[i, 0], future[i + 1, 0]],
                            [future[i, 1], future[i + 1, 1]],
                            color=colors[i], linewidth=FUTURE_LINE_WIDTH / 2, zorder=650)

                # Current Node Position
                circle = plt.Circle((history[-1, 0],
                                     history[-1, 1]),
                                    NODE_CIRCLE_SIZE,
                                    facecolor='g',
                                    edgecolor='k',
                                    lw=CIRCLE_EDGE_WIDTH,
                                    zorder=3)
                ax.add_artist(circle)

        if robot_plan is not None:
            future_plan = robot_plan[t][0:2, :]
            future_plan = future_plan.T

            # Create a custom color gradient from dark blue to light yellow
            colors = [plt.cm.viridis(i / len(future_plan)) for i in range(len(future_plan))]

            for i in range(np.size(future_plan, 0) - 1):
                # Draw the segment
                ax.plot([future_plan[i, 0], future_plan[i + 1, 0]],
                        [future_plan[i, 1], future_plan[i + 1, 1]],
                        color=colors[i], linewidth=FUTURE_LINE_WIDTH, zorder=650)

            # Draw the rectangle for the current position
            angle = robot_plan[t][2, 0] * 180 / np.pi - 90
            plot_rectangle(current_position=future_plan[0, :], angle=angle, ax=ax, color='blue', alpha=0.9)

        ax.set_aspect('equal')
        path = plot_path + 'Gaussian'
        if not os.path.isdir(path):
            os.mkdir(path)
        fig.savefig(path + '/t_' + str(t) + '.png', dpi=300, bbox_inches='tight')
        fig.savefig(path + '/t_' + str(t) + '.svg', bbox_inches='tight')


def plot_A(patch, layers, mats_outputs_collection, scene,
           nusc_map=None, max_hl=10, ph=6, pi_threshold=0.05,
           x_min=0, y_min=0,
           robot_plan=None, scene_num=None,
           plot_path=None):
    plot_path = plot_path + '/scene_' + str(scene_num) + '/'
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    for t in range(len(mats_outputs_collection)):
        fig, ax = nusc_map.render_map_patch(patch, layers, figsize=(23, 15), alpha=0.06, render_egoposes_range=False)

        # get prediction results
        prediction_distributions, _, As, _, _, _, state_lengths_in_order = mats_outputs_collection[t]

        prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_distributions,
                                                                                          max_hl,
                                                                                          ph,
                                                                                          map=None)

        assert (len(prediction_dict.keys()) <= 1)
        if len(prediction_dict.keys()) == 0:
            print("t", t)
        ts_key = list(prediction_dict.keys())[0]
        prediction_dict = prediction_dict[ts_key]
        histories_dict = histories_dict[ts_key]
        futures_dict = futures_dict[ts_key]

        " ------------Plot A Matrix----------------- "
        # get A matrix
        random_dist = next(iter(prediction_distributions[ts_key].values()))
        pis = random_dist.pis
        # index of the most likely mode
        ml_pi_idx = torch.argmax(pis).item()

        A = As[0]
        summed_state_lengths_in_order = torch.zeros(
            (state_lengths_in_order.shape[0], state_lengths_in_order.shape[1] + 1),
            dtype=state_lengths_in_order.dtype)
        summed_state_lengths_in_order[:, 1:] = torch.cumsum(state_lengths_in_order, dim=1)

        ordered_nodes = [scene.robot] + list(prediction_distributions[ts_key].keys())
        relation_magnitudes = np.zeros((A.shape[0], len(ordered_nodes), len(ordered_nodes)))
        for ts in trange(A.shape[0]):
            for orig_idx, orig_node in enumerate(ordered_nodes):
                for dest_idx, dest_node in enumerate(ordered_nodes):
                    if orig_idx == dest_idx:
                        continue

                    rows, cols = calculate_A_slices(orig_idx, dest_idx,
                                                    state_lengths_in_order[0],
                                                    summed_state_lengths_in_order[0])

                    relation_magnitudes[ts, dest_idx, orig_idx] = np.linalg.norm(A[ts, 0, ml_pi_idx, rows, cols])

        normed_magnitudes = relation_magnitudes / (relation_magnitudes.sum(axis=2, keepdims=True) + 1e-6)
        # average value in the following [ph] steps
        mean_normed_magnitude = np.linalg.norm(normed_magnitudes[:, :, 0], axis=0)
        mean_normed_magnitude = mean_normed_magnitude - mean_normed_magnitude.min()
        mean_normed_magnitude = mean_normed_magnitude / mean_normed_magnitude.max()
        max_normed_magnitude = np.max(normed_magnitudes[:, :, 0], axis=0)
        max_normed_magnitude = max_normed_magnitude - max_normed_magnitude.min()
        max_normed_magnitude = max_normed_magnitude / max_normed_magnitude.max()

        fig_A, ax_A = plt.subplots(figsize=(10, 8))
        sns.heatmap(normed_magnitudes[0], cmap='YlOrRd', vmin=0.0, vmax=1.0,
                    annot=True, cbar=True, square=True,
                    fmt=".2f", ax=ax_A)

        path = plot_path + 'A_Matrix'
        if not os.path.isdir(path):
            os.mkdir(path)
        fig_A.savefig(path + '/t_' + str(t) + '.png', dpi=300)
        fig_A.savefig(path + '/t_' + str(t) + '.svg')

        " ------------Plot A Connections----------------- "
        # current node list
        node_list = sorted(histories_dict.keys(), key=lambda x: x.id)

        future_plan = robot_plan[t][0:2, :]
        future_plan = future_plan.T

        position_state = {'position': ['x', 'y']}
        for orig_idx, orig_node in enumerate(ordered_nodes):
            for dest_idx, dest_node in enumerate(ordered_nodes):

                if orig_idx == dest_idx:
                    continue

                curr_orig = orig_node.get(np.array([ts_key]), position_state) + np.array([x_min, y_min])
                curr_dest = dest_node.get(np.array([ts_key]), position_state) + np.array([x_min, y_min])

                if orig_idx == 0:
                    curr_orig = future_plan
                if dest_idx == 0:
                    curr_dest = future_plan

                # Check if the node is in the ROI
                patch_x_min, patch_y_min, patch_x_max, patch_y_max = patch
                if (patch_x_min < curr_orig[0, 0] < patch_x_max and patch_y_min < curr_orig[0, 1] < patch_y_max and
                        patch_x_min < curr_dest[0, 0] < patch_x_max and patch_y_min < curr_dest[0, 1] < patch_y_max):

                    line_weight = normed_magnitudes[0, dest_idx, orig_idx]
                    if line_weight < 0.05:
                        line_weight = 0
                    else:
                        line_weight = line_weight * 2

                    ax.plot([curr_orig[0, 0], curr_dest[0, 0]],
                            [curr_orig[0, 1], curr_dest[0, 1]],
                            c='k', linewidth=line_weight, alpha=0.7 * line_weight)

        " ------------Plot Agents----------------- "
        for node_idx, node in enumerate(ordered_nodes):
            if node_idx == 0:
                continue
            history = histories_dict[node] + np.array([x_min, y_min])
            future = futures_dict[node] + np.array([x_min, y_min])
            predictions = prediction_dict[node]
            pis = predictions.pis  # the possibility of a mode

            if node.type.name == 'VEHICLE':

                # Future
                # Create a custom color gradient from dark blue to light yellow
                colors = [plt.cm.viridis(i / len(future)) for i in range(len(future))]

                for i in range(np.size(future, 0) - 1):
                    # Draw the segment
                    ax.plot([future[i, 0], future[i + 1, 0]],
                            [future[i, 1], future[i + 1, 1]],
                            color=colors[i], linewidth=FUTURE_LINE_WIDTH / 2, zorder=650, alpha=0.5)

                # Current position
                alpha_weight = mean_normed_magnitude[node_idx]
                angle = node.get(np.array([ts_key]), {'heading': ['°']})[0, 0] * 180 / np.pi - 90
                plot_rectangle(current_position=history[-1, :], angle=angle, ax=ax, color='red', alpha=0.7 * alpha_weight)

            elif node.type.name in {'PEDESTRIAN'}:

                # Future
                # Create a custom color gradient from dark blue to light yellow
                colors = [plt.cm.viridis(i / len(future)) for i in range(len(future))]

                for i in range(np.size(future, 0) - 1):
                    # Draw the segment
                    ax.plot([future[i, 0], future[i + 1, 0]],
                            [future[i, 1], future[i + 1, 1]],
                            color=colors[i], linewidth=FUTURE_LINE_WIDTH / 2, zorder=650, alpha=0.5)

                # Current Node Position
                alpha_weight = mean_normed_magnitude[node_idx]
                circle = plt.Circle((history[-1, 0],
                                     history[-1, 1]),
                                    NODE_CIRCLE_SIZE,
                                    facecolor='red',
                                    edgecolor='k',
                                    lw=CIRCLE_EDGE_WIDTH,
                                    zorder=3, alpha=0.7 * alpha_weight)
                ax.add_artist(circle)

        if robot_plan is not None:
            future_plan = robot_plan[t][0:2, :]
            future_plan = future_plan.T

            # Create a custom color gradient from dark blue to light yellow
            colors = [plt.cm.viridis(i / len(future_plan)) for i in range(len(future_plan))]

            for i in range(np.size(future_plan, 0) - 1):
                # Draw the segment
                ax.plot([future_plan[i, 0], future_plan[i + 1, 0]],
                        [future_plan[i, 1], future_plan[i + 1, 1]],
                        color=colors[i], linewidth=FUTURE_LINE_WIDTH, zorder=650)

            # Draw the rectangle for the current position
            angle = robot_plan[t][2, 0] * 180 / np.pi - 90
            plot_rectangle(current_position=future_plan[0, :], angle=angle, ax=ax, color='blue', alpha=0.9)

        ax.set_aspect('equal')

        path = plot_path + 'A_Connection'
        if not os.path.isdir(path):
            os.mkdir(path)
        fig.savefig(path + '/t_' + str(t) + '.png', dpi=300, bbox_inches='tight')
        fig.savefig(path + '/t_' + str(t) + '.svg', bbox_inches='tight')


def plot_B(patch, layers, mats_outputs_collection, scene,
           nusc_map=None, max_hl=10, ph=6, pi_threshold=0.05,
           x_min=0, y_min=0,
           robot_plan=None, scene_num=None,
           plot_path=None):
    plot_path = plot_path + '/scene_' + str(scene_num) + '/'
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    for t in range(len(mats_outputs_collection)):
        fig, ax = nusc_map.render_map_patch(patch, layers, figsize=(23, 15), alpha=0.06, render_egoposes_range=False)

        # get prediction results
        prediction_distributions, _, As, Bs, _, _, state_lengths_in_order = mats_outputs_collection[t]

        prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_distributions,
                                                                                          max_hl,
                                                                                          ph,
                                                                                          map=None)

        assert (len(prediction_dict.keys()) <= 1)
        if len(prediction_dict.keys()) == 0:
            print("t", t)
        ts_key = list(prediction_dict.keys())[0]
        prediction_dict = prediction_dict[ts_key]
        histories_dict = histories_dict[ts_key]
        futures_dict = futures_dict[ts_key]

        " ------------Plot B Matrix----------------- "
        # get B matrix
        random_dist = next(iter(prediction_distributions[ts_key].values()))
        pis = random_dist.pis
        ml_pi_idx = torch.argmax(pis).item()

        B = Bs[0]
        summed_state_lengths_in_order = torch.zeros(
            (state_lengths_in_order.shape[0], state_lengths_in_order.shape[1] + 1),
            dtype=state_lengths_in_order.dtype)
        summed_state_lengths_in_order[:, 1:] = torch.cumsum(state_lengths_in_order, dim=1)

        ordered_nodes = [scene.robot] + list(prediction_distributions[ts_key].keys())
        relation_magnitudes = np.zeros((B.shape[0], len(ordered_nodes)))
        for ts in trange(B.shape[0]):
            for dest_idx, dest_node in enumerate(ordered_nodes):
                if dest_idx == 0:
                    continue

                rows = calculate_BQ_slices(dest_idx,
                                           state_lengths_in_order[0],
                                           summed_state_lengths_in_order[0])

                relation_magnitudes[ts, dest_idx] = np.linalg.norm(B[ts, 0, ml_pi_idx, rows])

        normed_magnitudes = relation_magnitudes[:, 1:] / relation_magnitudes.max()
        # average value in the following [ph] steps
        mean_normed_magnitude = np.linalg.norm(normed_magnitudes, axis=0)
        max_normed_magnitude = np.max(normed_magnitudes, axis=0)
        max_normed_magnitude = max_normed_magnitude - max_normed_magnitude.min()
        max_normed_magnitude = max_normed_magnitude / max_normed_magnitude.max()

        fig_B, ax_B = plt.subplots(figsize=(8, 16))
        line_locs = range(1, ph)
        sns.heatmap(normed_magnitudes.T, cmap='YlOrRd',
                    vmin=0.0, vmax=1.0,
                    annot=True, cbar=True, square=True,
                    fmt=".2f", ax=ax_B,
                    xticklabels=range(1, ph + 1),
                    yticklabels=range(1, len(ordered_nodes)))

        ax_B.vlines(line_locs, *(ax_B.get_ylim()), colors=['white'], linewidths=3)
        ax_B.set_xlabel('Prediction Timestep')
        ax_B.set_ylabel('Agent Number')

        path = plot_path + 'B_Matrix'
        if not os.path.isdir(path):
            os.mkdir(path)
        fig_B.savefig(path + '/t_' + str(t) + '.png', dpi=600)
        fig_B.savefig(path + '/t_' + str(t) + '.svg')

        " ------------Plot B Weight----------------- "
        # current node list
        node_list = sorted(histories_dict.keys(), key=lambda x: x.id)

        " ------------Plot Agents----------------- "
        for node_idx, node in enumerate(ordered_nodes):
            if node_idx == 0:
                continue
            history = histories_dict[node] + np.array([x_min, y_min])
            future = futures_dict[node] + np.array([x_min, y_min])
            predictions = prediction_dict[node]
            pis = predictions.pis  # the possibility of a mode

            if node.type.name == 'VEHICLE':

                # Future
                # Create a custom color gradient from dark blue to light yellow
                colors = [plt.cm.viridis(i / len(future)) for i in range(len(future))]

                for i in range(np.size(future, 0) - 1):
                    # Draw the segment
                    ax.plot([future[i, 0], future[i + 1, 0]],
                            [future[i, 1], future[i + 1, 1]],
                            color=colors[i], linewidth=FUTURE_LINE_WIDTH / 2, zorder=650, alpha=0.5)

                # Current position
                alpha_weight = max_normed_magnitude[node_idx - 1]
                angle = node.get(np.array([ts_key]), {'heading': ['°']})[0, 0] * 180 / np.pi - 90
                plot_rectangle(current_position=history[-1, :], angle=angle, ax=ax, color='red',
                               alpha=0.7 * alpha_weight)

            elif node.type.name in {'PEDESTRIAN'}:

                # Future
                # Create a custom color gradient from dark blue to light yellow
                colors = [plt.cm.viridis(i / len(future)) for i in range(len(future))]

                for i in range(np.size(future, 0) - 1):
                    # Draw the segment
                    ax.plot([future[i, 0], future[i + 1, 0]],
                            [future[i, 1], future[i + 1, 1]],
                            color=colors[i], linewidth=FUTURE_LINE_WIDTH / 2, zorder=650, alpha=0.5)

                # Current Node Position
                alpha_weight = max_normed_magnitude[node_idx - 1]
                circle = plt.Circle((history[-1, 0],
                                     history[-1, 1]),
                                    NODE_CIRCLE_SIZE,
                                    facecolor='red',
                                    edgecolor='k',
                                    lw=CIRCLE_EDGE_WIDTH,
                                    alpha=0.7 * alpha_weight,
                                    zorder=3)
                ax.add_artist(circle)

        if robot_plan is not None:
            future_plan = robot_plan[t][0:2, :]
            future_plan = future_plan.T

            # Create a custom color gradient from dark blue to light yellow
            colors = [plt.cm.viridis(i / len(future_plan)) for i in range(len(future_plan))]

            for i in range(np.size(future_plan, 0) - 1):
                # Draw the segment
                ax.plot([future_plan[i, 0], future_plan[i + 1, 0]],
                        [future_plan[i, 1], future_plan[i + 1, 1]],
                        color=colors[i], linewidth=FUTURE_LINE_WIDTH, zorder=650)

            # Draw the rectangle for the current position
            angle = robot_plan[t][2, 0] * 180 / np.pi - 90
            plot_rectangle(current_position=future_plan[0, :], angle=angle, ax=ax, color='blue', alpha=0.9)

        ax.set_aspect('equal')

        path = plot_path + 'B_Weighted'
        if not os.path.isdir(path):
            os.mkdir(path)
        fig.savefig(path + '/t_' + str(t) + '.png', dpi=600, bbox_inches='tight')
        fig.savefig(path + '/t_' + str(t) + '.svg', bbox_inches='tight')


def plot_node_Gaussian_distribution(means, covs, ax, possibility):
    ax.plot(means[..., 0], means[..., 1],
            '-o', markersize=1,
            color=cmap[1],
            linewidth=LINE_WIDTH,
            alpha=LINE_ALPHA,
            zorder=620)

    for timestep in range(means.shape[0]):
        mean = means[timestep]
        covar = covs[timestep]

        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(u[1], u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = patches.Ellipse(mean, v[0], v[1], 180. + angle, color='blue', zorder=600)
        ell.set_edgecolor(None)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(possibility / 8.)
        ax.add_artist(ell)


def plot_rectangle(current_position, angle, ax=None, alpha=0.7, color='gray'):

    # rect = patches.Rectangle((history[-1, 0] - VEHICLE_WIDTH / 2, history[-1, 1] - VEHICLE_HEIGHT / 2),
    #                          VEHICLE_WIDTH, VEHICLE_HEIGHT, color='gray', alpha=0.7)

    # Create the rectangle (its initial position doesn't matter as we'll transform it)
    rect = patches.Rectangle((0, 0), VEHICLE_WIDTH, VEHICLE_HEIGHT, color=color, alpha=alpha)

    # Create a transformation for the rotation and translation
    # First, translate the rectangle to the origin (0, 0), rotate, and then translate to the desired position
    trans = transforms.Affine2D().translate(-VEHICLE_WIDTH / 2, -VEHICLE_HEIGHT / 2).rotate_deg(angle).translate(*current_position)

    # Combine this transformation with the data coordinate transformation
    combined_trans = trans + ax.transData

    # Apply the combined transformation to the rectangle
    rect.set_transform(combined_trans)

    # Add the rectangle to the plot
    ax.add_patch(rect)
    rect.zorder = 700
    ax.add_patch(rect)
