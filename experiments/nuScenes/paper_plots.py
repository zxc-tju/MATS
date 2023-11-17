import sys
import os
import time

import dill
import glob
import json
import random
import argparse
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.append("../../mats")
from utils import prediction_output_to_trajectories
from tqdm import tqdm, trange
from model.model_registrar import ModelRegistrar
from model.mats import MATS
from environment.node import MultiNode
from model.dataset import EnvironmentDataset
import evaluation
import visualization
import helper as plotting_helper

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    mats = MATS(model_registrar, hyperparams, None, 'cpu')
    return mats, hyperparams


def load_latest_model(model_dir, env):
    latest_model_path = sorted(glob.glob(os.path.join(model_dir, '*.pt')))[-1]
    latest_model_iter = int(latest_model_path.split('-')[-1].split('.')[0])
    return load_model(model_dir, env, ts=latest_model_iter)


def visualize_mat(mat, pred_dists, state_lengths_in_order, ax,
                  vmin=-0.0001, center=0.00, vmax=0.0001):
    timesteps, num_samples, components = mat.shape[:3]

    random_dist = next(iter(pred_dists.values()))
    pis = random_dist.pis
    ml_pi_idx = torch.argmax(pis).item()

    line_locs = state_lengths_in_order.cumsum(1)

    sns.heatmap(mat[0, 0, ml_pi_idx].cpu(),
                vmin=vmin, center=center, vmax=vmax, cmap='coolwarm',
                annot=False, cbar=False, square=True,
                fmt=".2f", ax=ax)
    #     ax.set_title('P(z=%d | x) = %.2f' % (ml_pi_idx, pis[ml_pi_idx]))
    ax.hlines(line_locs, *(ax.get_xlim()), colors=['white'], linewidths=3)
    ax.vlines(line_locs, *(ax.get_ylim()), colors=['white'], linewidths=3)
    ax.axis('off')


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

nuScenes_data_path = '/home/zxc/codes/MATS/experiments/nuScenes/data'
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

with open('../processed/nuScenes_val_full.pkl', 'rb') as f:
    env = dill.load(f, encoding='latin1')

# Modeling Loading
mats, hyperparams = load_model('models/models_26_Jul_2020_17_11_34_full_zeroRrows_batch8_fixed_edges',
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

for scene in tqdm(scenes):
    scene.calculate_scene_graph(env.attention_radius,
                                hyperparams['edge_addition_filter'],
                                hyperparams['edge_removal_filter'])

ph = hyperparams['prediction_horizon']
max_hl = hyperparams['maximum_history_length']


scene = scenes[5]  # np.random.choice(scenes)
timestep = np.array([30])  # scene.sample_timesteps(1, min_future_timesteps=ph, min_history_length=max_hl)

# Get Prediction Results
mats_outputs = list()
with torch.no_grad():
    mats_outputs.append(mats.predict(scene,
                                     timestep,
                                     ph,
                                     min_future_timesteps=ph,
                                     include_B=hyperparams['include_B'],
                                     zero_R_rows=hyperparams['zero_R_rows']))

pred_dists, non_rob_rows, As, Bs, Qs, affine_terms, state_lengths_in_order = mats_outputs[0]
prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(pred_dists,
                                                                                  max_hl,
                                                                                  ph,
                                                                                  map=None)

# Define ROI in nuScenes Map
nusc_map = NuScenesMap(dataroot=nuScenes_data_path, map_name=helper.get_map_name_from_sample_token(scene.name))

ts_key = list(prediction_dict.keys())[0]
histories_dict = histories_dict[ts_key]
node = list(histories_dict.keys())[0]
histories_one_step = histories_dict[node][0]
x_min = scene.x_min + histories_one_step[0] - 75
y_min = scene.y_min + histories_one_step[1] - 75
x_max = scene.x_min + histories_one_step[0] + 75.0
y_max = scene.y_min + histories_one_step[1] + 75.0

for idx in trange(len(mats_outputs)):
    pred_dists, non_rob_rows, As, Bs, Qs, affine_terms, state_lengths_in_order = mats_outputs[idx]

    # Plot predicted timestep for random scene in map
    my_patch = (x_min, y_min, x_max, y_max)

    fig, ax = nusc_map.render_map_patch(my_patch, layers, figsize=(7, 7), alpha=0.1, render_egoposes_range=False)

    # Plot predicted timestep for random scene
    # visualization.visualize_prediction(ax,
    #                                    pred_dists,
    #                                    scene.dt,
    #                                    max_hl=max_hl,
    #                                    ph=ph,
    #                                    map=None,
    #                                    robot_node=scene.robot,
    #                                    x_min=scene.x_min,
    #                                    y_min=scene.y_min)

    plotting_helper.plot_vehicle_dist(ax,
                                      pred_dists,
                                      scene,
                                      max_hl=max_hl,
                                      ph=ph,
                                      x_min=scene.x_min,
                                      y_min=scene.y_min,
                                      line_width=0.5,
                                      car_img_zoom=0.02)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
plt.show()
fig.savefig('plots/scene_predictions_dist_2.pdf', dpi=300, bbox_inches='tight')
