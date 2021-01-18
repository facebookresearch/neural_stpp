# Copyright (c) Facebook, Inc. and its affiliates.

from tqdm import tqdm
import pandas as pd
import numpy as np
import reverse_geocoder as rg

from download_utils import download_url


def main():
    np.random.seed(0)

    download_url("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv", "covid19")

    df = pd.read_csv("covid19/us-counties.csv")

    # Remove rows with unknown counties.
    df = df[df.county != "Unknown"]

    # Merge with latitude and longitude.
    url = "https://en.m.wikipedia.org/wiki/User:Michael_J/County_table"
    df_county_geoloc = pd.read_html(url)[0]
    df_county_geoloc["Area"] = df_county_geoloc["Total Areakm²"]
    df_county_geoloc = df_county_geoloc[["FIPS", "Longitude", "Latitude", "Area"]]

    df_county_geoloc.Latitude = df_county_geoloc.Latitude.map(lambda s: float(s.replace("–", "-").replace("°", "")))
    df_county_geoloc.Longitude = df_county_geoloc.Longitude.map(lambda s: float(s.replace("–", "-").replace("°", "")))
    df_county_geoloc.FIPS = df_county_geoloc.FIPS.map(lambda x: float(x))
    df = df.merge(df_county_geoloc, left_on="fips", right_on="FIPS", how="left")

    # Fill in rows with NaN FIPS.
    df.set_index("county", inplace=True)
    missing_latlong = [
        ["New York City", 40.7128, -74.0060, 783.8],
        ["Kansas City", 39.0997, -94.5786, 815.72],
        ["Joplin", 37.0842, -94.5133, 81.7],
        ["Kusilvak Census Area", 62.1458, -162.8919, 44240],
    ]

    df_na = pd.DataFrame(missing_latlong, columns=["county", "Longitude", "Latitude", "Area"])
    df_na.set_index("county", inplace=True)
    df.update(df_na, overwrite=False)
    df = df.reset_index()

    # Remove Alaska and Hawaii.
    df = df[df.state != "Alaska"]
    df = df[df.state != "Hawaii"]

    # Compute number of new cases/deaths each day instead of cumulative.
    df.sort_values(by=["state", "county", "date"], inplace=True)

    df["new_cases"] = df.groupby(["state", "county"])["cases"].diff().fillna(df["cases"])
    df["new_deaths"] = df.groupby(["state", "county"])["deaths"].diff().fillna(df["deaths"])

    # Select time line from March to June.
    df["date"] = pd.to_datetime(df["date"])
    start_date = pd.Timestamp("2020-03-15")
    end_date = pd.Timestamp("2020-08-01")
    df = df[pd.DatetimeIndex(df.date) >= start_date]
    df = df[pd.DatetimeIndex(df.date) <= end_date]

    # Create numeric time column.
    df["day"] = df["date"].apply(lambda x: float((x - start_date).days))

    # Cases in New Jersey.
    df = df[["day", "Longitude", "Latitude", "Area", "new_cases", "state", "county"]]
    df = df[df.new_cases > 0]
    df = df.loc[df.index.repeat(df.new_cases)]
    df = df[df.state == "New Jersey"]

    # Break into 7 day intervals using a sliding window of 3 days.
    sequences = {}
    interval_length = 7
    for start in range(0, int(df["day"].max()) - interval_length + 1, 3):
        date = start_date + pd.Timedelta(days=start)
        seq_name = f"{date.year}{date.month:02d}" + f"{date.day:02d}"

        df_range = df[df["day"] >= start]
        df_range = df_range[df_range["day"] < start + interval_length]
        df_range["day"] = df_range["day"] - start

        seq = df_range.to_numpy()[:, :4].astype(np.float64)
        counties = df_range.to_numpy()[:, -1]

        t, x = seq[:, 0:1], seq[:, 1:3]
        area = seq[:, 3]

        print(seq_name, seq.shape[0])

        for i in tqdm(range(50)):
            # subsample_idx = np.sort(np.random.choice(seq.shape[0], seq.shape[0] // 200, replace=False))
            subsample_idx = np.random.rand(seq.shape[0]) < (1 / 100)

            while np.sum(subsample_idx) == 0:
                subsample_idx = np.random.rand(seq.shape[0]) < (1 / 100)

            # Uniformly distribute the daily case count.
            _t = add_temporal_noise(t[subsample_idx])

            # Assume each degree of longitude/latitude is ~110km.
            degrees = np.sqrt(area) / 110.0
            _x = add_unif_spatial_noise(x[subsample_idx], degrees[subsample_idx].reshape(-1, 1), counties[subsample_idx])

            sort_idx = np.argsort(_t.reshape(-1))
            sequences[seq_name + f"_{i:03d}"] = np.concatenate([_t, _x], axis=1)[sort_idx]

    np.savez("covid19/covid_nj_cases.npz", **sequences)


def add_unif_spatial_noise(coords, width, counties):
    sampled_coords = coords

    match = np.zeros(sampled_coords.shape[0]) > 0
    while not match.all():
        sampled_coords = sampled_coords * match.reshape(-1, 1) + (coords + 2.0 * (np.random.rand(*coords.shape) * width - width / 2)) * ~match.reshape(-1, 1)
        lons, lats = sampled_coords[:, 0], sampled_coords[:, 1]
        queries = list(zip(lats, lons))
        results = rg.search(queries)
        match = np.array([county in res["admin2"] for county, res in zip(counties, results)])

    return sampled_coords


def add_temporal_noise(day):
    return day + np.random.rand(*day.shape)


if __name__ == "__main__":
    main()
