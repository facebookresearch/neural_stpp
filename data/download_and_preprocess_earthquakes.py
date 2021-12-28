# Copyright (c) Facebook, Inc. and its affiliates.

import pandas as pd
import numpy as np
import urllib.request
import shutil
import os


def download_data(year_start, year_end):

    url = ("https://earthquake.usgs.gov/fdsnws/event/1/query.csv?"
           f"starttime={year_start}-01-01%2000:00:00&endtime={year_end}"
           "-12-31%2023:59:59&maxlatitude=46&minlatitude=22&maxlongitude=150"
           "&minlongitude=122&minmagnitude=2.5&eventtype=earthquake&orderby=time-asc")

    filename = f"earthquakes/{year_start}_{year_end}.csv"
    os.makedirs("earthquakes", exist_ok=True)
    urllib.request.urlretrieve(url, filename)
    return filename


def preprocess():
    df = pd.read_csv("earthquakes/1990_2019.csv")

    # filter earthquakes in Japan
    df = df[df["place"].str.contains("Japan")]
    df = df[["time", "longitude", "latitude"]]

    # write time as a days since 2010-01-01
    basedate = pd.Timestamp(("1990-01-01T00:00:00Z"))
    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"].apply(lambda x: (x - basedate).total_seconds() / 60 / 60 / 24)  # days

    sequences = {}
    for weeks in range(52 * 30 + 1):
        date = basedate + pd.Timedelta(weeks=weeks)

        if date.year == 2011:
            continue
        if date.year == 2010 and date.month in [11, 12]:
            continue

        seq_name = f"{date.year}{date.month:02d}" + f"{date.day:02d}"

        start = (date - basedate).days
        end = (date + pd.Timedelta(days=30) - basedate).days

        df_range = df[df["time"] >= start]
        df_range = df_range[df_range["time"] < end]
        df_range["time"] = df_range["time"] - start

        seq = df_range.to_numpy().astype(np.float64)

        t, x = seq[:, 0:1], seq[:, 1:3]

        sequences[seq_name] = np.concatenate([t, x], axis=1)

    np.savez("earthquakes/earthquakes_jp.npz", **sequences)
    print("Preprocessing complete.")


if __name__ == "__main__":

    filenames = []
    filenames.append(download_data(1990, 1999))
    filenames.append(download_data(2000, 2009))
    filenames.append(download_data(2010, 2019))

    # combine csv files
    with open('earthquakes/1990_2019.csv', 'wb') as wfd:
        for i, f in enumerate(filenames):
            with open(f, 'rb') as fd:
                if i != 0:
                    fd.readline()  # Throw away header on all but first file
                shutil.copyfileobj(fd, wfd)

    preprocess()
