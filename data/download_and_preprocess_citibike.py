# Copyright (c) Facebook, Inc. and its affiliates.

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import zipfile
from download_utils import download_url


def download(root="", year=2019):
    for month in range(1, 13):
        url = f"https://s3.amazonaws.com/tripdata/{year}{month:02d}-citibike-tripdata.csv.zip"
        dirname = os.path.join(root, "citibike", "raw")
        if download_url(url, os.path.join(root, "citibike", "raw")):
            filename = os.path.join(dirname, os.path.basename(url))
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(dirname)


def process(root="", year=2019):

    np.random.seed(0)

    dfs = []
    for month in tqdm(range(1, 13)):
        filepath = os.path.join(root, "citibike", "raw", f"{year}{month:02d}-citibike-tripdata.csv")
        dfs.append(pd.read_csv(filepath))
    df = pd.concat(dfs)

    stds = df.std(0)
    std_x = stds["start station longitude"]
    std_y = stds["start station latitude"]

    df["starttime"] = pd.to_datetime(df["starttime"])
    timedelta = pd.Timedelta(hours=5)
    df["starttime"] = df["starttime"] - timedelta

    df_by_day = []
    for group in df.groupby(pd.DatetimeIndex(df["starttime"]).date):
        df_by_day.append(group[1])

    sequences = {}
    for _, df in enumerate(df_by_day[3:-1]):

        starttime = df["starttime"]

        year = pd.DatetimeIndex(starttime).year[0]
        month = pd.DatetimeIndex(starttime).month[0]
        day = pd.DatetimeIndex(starttime).day[0]

        t = (
            pd.DatetimeIndex(starttime).hour * 60 * 60 +
            pd.DatetimeIndex(starttime).minute * 60 +
            pd.DatetimeIndex(starttime).second +
            pd.DatetimeIndex(starttime).microsecond * 1e-6
        )
        t = np.array(t) / 60 / 60

        y = df["start station latitude"]
        x = df["start station longitude"]

        x = np.array(x)
        y = np.array(y)

        seq = np.stack([t, x, y], axis=1)
        seq_name = f"{year}{month:02d}" + f"{day:02d}"

        print(seq_name)

        for i in range(20):
            # subsample_idx = np.sort(np.random.choice(seq.shape[0], seq.shape[0] // 500, replace=False))
            subsample_idx = np.random.rand(seq.shape[0]) < (1 / 500)
            while np.sum(subsample_idx) == 0:
                subsample_idx = np.random.rand(seq.shape[0]) < (1 / 500)

            sequences[seq_name + f"_{i:03d}"] = add_spatial_noise(seq[subsample_idx], std=[0., std_x * 0.02, std_y * 0.02])

            print(np.sum(subsample_idx))

    np.savez("citibike/citibike2.npz", **sequences)


def add_spatial_noise(coords, std=[0., 0., 0.]):
    return coords + np.random.randn(*coords.shape) * std


if __name__ == "__main__":
    download()
    process()
