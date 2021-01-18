# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os
import matplotlib
matplotlib.use("Agg")  # noqa
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from scipy.stats import gaussian_kde
import torch

import datasets
import toy_datasets

BBOXES = {
    "citibike": (-74.03, -73.87, 40.65, 40.87),
    "covid_nj_cases": (-75.60, -73.90, 38.90, 41.20),
    "earthquakes_jp": (123.43, 149.18, 25.41, 45.98),
    "pinwheel": (-4.0, 4.0, -4.0, 4.0),
    "fmri": (0.0, 106.0, 0.0, 106.0),
}

MAPS = {
    "citibike": "assets/manhattan_map.png",
    "covid_nj_cases": "assets/nj_map.png",
    "earthquakes_jp": "assets/jp_map.png",
    "pinwheel": None,
    "fmri": None,
    "gmm": None,
}

FIGSIZE = 10
DPI = 300


def plot_coordinates(coords, S_std, S_mean, savepath, dataset_name):
    coords = coords * S_std + S_mean
    longs = coords[:, 0].detach().cpu().numpy()
    lats = coords[:, 1].detach().cpu().numpy()

    if MAPS[dataset_name]:
        map_img = plt.imread(MAPS[dataset_name])
        fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE * map_img.shape[0] / map_img.shape[1]))
        ax.imshow(map_img, zorder=0, extent=BBOXES[dataset_name])
    else:
        fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE))

    ax.scatter(longs, lats, s=1, alpha=0.4)
    ax.set_xlim(BBOXES[dataset_name][0], BBOXES[dataset_name][1])
    ax.set_ylim(BBOXES[dataset_name][2], BBOXES[dataset_name][3])

    plt.axis('off')
    os.makedirs(os.path.join(savepath, f"{dataset_name}"), exist_ok=True)
    plt.savefig(os.path.join(savepath, f"{dataset_name}", f"{dataset_name}.png"), bbox_inches='tight', dpi=DPI)
    plt.close()


def plot_kde(coords, S_std, S_mean, savepath, dataset_name, text=None, name=None):
    name = f"{dataset_name}_density" if name is None else name

    coords = coords * S_std.to(coords) + S_mean.to(coords)
    longs = coords[:, 0].detach().cpu().numpy()
    lats = coords[:, 1].detach().cpu().numpy()

    if MAPS[dataset_name]:
        map_img = plt.imread(MAPS[dataset_name])
        fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE * map_img.shape[0] / map_img.shape[1]))
        ax.imshow(map_img, zorder=0, extent=BBOXES[dataset_name])
    else:
        fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE))

    kernel = gaussian_kde(np.stack([longs, lats], axis=0))
    kernel.inv_cov = np.diag(np.diag(kernel.inv_cov))
    X, Y = np.mgrid[BBOXES[dataset_name][0]:BBOXES[dataset_name][1]:100j, BBOXES[dataset_name][2]:BBOXES[dataset_name][3]:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)
    ax.contourf(X, Y, Z, levels=10, alpha=0.6, cmap='RdGy')
    ax.set_xlim(BBOXES[dataset_name][0], BBOXES[dataset_name][1])
    ax.set_ylim(BBOXES[dataset_name][2], BBOXES[dataset_name][3])

    if text is not None:
        txt = ax.text(0.15, 0.9, text,
                      horizontalalignment="center",
                      verticalalignment="center",
                      transform=ax.transAxes,
                      size=16,
                      color='white')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='black')])

    plt.axis('off')
    os.makedirs(os.path.join(savepath, f"{dataset_name}"), exist_ok=True)
    plt.savefig(os.path.join(savepath, f"{dataset_name}", f"{name}.png"), bbox_inches='tight', dpi=DPI)
    plt.close()


def plot_density(loglik_fn, spatial_locations, index, S_mean, S_std, savepath, dataset_name, device, text=None, fp64=False):
    N = 50

    x = np.linspace(BBOXES[dataset_name][0], BBOXES[dataset_name][1], N)
    y = np.linspace(BBOXES[dataset_name][2], BBOXES[dataset_name][3], N)
    s = np.stack([x, y], axis=1)

    X, Y = np.meshgrid(s[:, 0], s[:, 1])
    S = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    S = torch.tensor(S).to(device)
    S = S.double() if fp64 else S.float()
    S = (S - S_mean.to(S)) / S_std.to(S)
    logp = loglik_fn(S)

    if MAPS[dataset_name]:
        map_img = plt.imread(MAPS[dataset_name])
        fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE * map_img.shape[0] / map_img.shape[1]))
        ax.imshow(map_img, zorder=0, extent=BBOXES[dataset_name])
    else:
        fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE))

    Z = logp.exp().detach().cpu().numpy().reshape(N, N)
    ax.contourf(X, Y, Z, levels=20, alpha=0.7, cmap='RdGy')

    spatial_locations = spatial_locations * np.array(S_std) + np.array(S_mean)
    ax.scatter(spatial_locations[:, 0], spatial_locations[:, 1], s=20**2, alpha=1.0, marker="x", color="k")

    ax.set_xlim(BBOXES[dataset_name][0], BBOXES[dataset_name][1])
    ax.set_ylim(BBOXES[dataset_name][2], BBOXES[dataset_name][3])

    if text:
        txt = ax.text(0.15, 0.9, text,
                      horizontalalignment="center",
                      verticalalignment="center",
                      transform=ax.transAxes,
                      size=16,
                      color='white')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='black')])

    plt.axis('off')
    os.makedirs(os.path.join(savepath, "figs"), exist_ok=True)
    np.savez(f"{savepath}/figs/data{index}.npz", **{"X": X, "Y": Y, "Z": Z, "spatial_locations": spatial_locations})
    plt.savefig(os.path.join(savepath, "figs", f"density{index}.png"), bbox_inches='tight', dpi=DPI)
    plt.close()


def plot_intensities(list_of_event_times, list_of_intensities, list_of_timevals, savepath):
    fig, axes = plt.subplots(nrows=len(list_of_event_times), figsize=(12, 1.5 * len(list_of_event_times)), sharex=True)
    for ax, event_times, intensities, timevals in zip(axes, list_of_event_times, list_of_intensities, list_of_timevals):
        ax.plot(timevals, intensities)
        ax.vlines(event_times, ymin=0.0, ymax=100.0, linestyles="--", linewidth=1, alpha=0.35)
        ax.set_xlim([timevals[0], timevals[-1]])
        ax.set_ylim([0., np.max(intensities) + 0.2])
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, bbox_inches='tight', dpi=DPI)
    plt.close()


def load_data(data, split="train"):

    if data == "citibike":
        return datasets.Citibike(split=split)
    elif data == "covid_nj_cases":
        return datasets.CovidNJ(split=split)
    elif data == "earthquakes_jp":
        return datasets.Earthquakes(split=split)
    elif data == "pinwheel":
        return toy_datasets.PinwheelHawkes(split=split)
    elif data == "gmm":
        return toy_datasets.GMMHawkes(split=split)
    elif data == "fmri":
        return datasets.BOLD5000(split=split)
    else:
        raise ValueError(f"Unknown data option {data}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, choices=MAPS.keys(), default="citibike")
    args = parser.parse_args()

    dataset = load_data(args.data)

    savepath = "dataset_figs"
    seq = dataset.__getitem__(0)
    event_times, spatial_locations = seq[:, 0], seq[:, 1:]
    plot_coordinates(spatial_locations, S_mean=dataset.S_mean, S_std=dataset.S_std, savepath=savepath, dataset_name=args.data)
    plot_kde(spatial_locations, S_mean=dataset.S_mean, S_std=dataset.S_std, savepath=savepath, dataset_name=args.data)
