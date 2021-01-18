# Copyright (c) Facebook, Inc. and its affiliates.

# Download dataset from
# https://bold5000.github.io/
# and place folders in data/bold5000


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import glob
import re
import pydicom
from tqdm import tqdm
import numpy as np

threshold = 6.0
np.random.seed(0)

sequences = {}
for dirname in tqdm(glob.glob("bold5000/*Unfilt_BOLD_CSI1_Sess*")):
    m = re.match(r".*Sess-(\d+)_Run-(\d+)", dirname)
    sess = int(m.group(1))
    run = int(m.group(2))
    events = []

    voxels = []
    for i, filename in enumerate(sorted(glob.glob(f"{dirname}/*.dcm"))):
        dataset = pydicom.dcmread(filename)
        voxel_t = dataset.pixel_array.reshape(9, 106, 9, 106).transpose(0, 2, 1, 3).reshape(81, 106, 106).transpose(1, 2, 0)
        voxels.append(voxel_t)

    # Get mean and standard deviation across time.
    voxels = np.stack(voxels, axis=0)
    mean = np.mean(voxels, axis=0)[None].repeat(voxels.shape[0], 0)
    std = np.std(voxels, axis=0)[None].repeat(voxels.shape[0], 0)

    z_scores = np.where(std > 0, (voxels - mean) / std, np.zeros(voxels.shape))

    event_indicator = (z_scores[:-1] < threshold) & (z_scores[1:] > threshold)

    events = np.transpose(np.nonzero(event_indicator))

    # Dequantize into uniform grid (0, 194)
    events = events + np.random.uniform(0, 1, size=events.shape)

    # Extract into ten sequences.
    split_times = np.linspace(0, 194, 11)
    for i, (t0, t1) in enumerate(zip(split_times[:-1], split_times[1:])):
        events_i = events[events[:, 0] >= t0]
        events_i = events_i[events_i[:, 0] < t1]
        events_i[:, 0] = events_i[:, 0] - t0
        events_i[:, 0] = events_i[:, 0] / (t1 - t0) * 10
        print(events_i.shape)
        sequences[f"{sess:02d}{run:02d}{i:02d}"] = events_i

    np.savez("bold5000/bold5000.npz", **sequences)
