# Copyright (c) Facebook, Inc. and its affiliates.

from functools import partial
import contextlib
import numpy as np

from datasets import SpatioTemporalDataset
from MHP import MHP

END_TIME = 30.0


class PinwheelHawkes(SpatioTemporalDataset):

    def __init__(self, split="train"):
        num_classes = 10
        m = np.array([0.05] * num_classes)
        a = np.diag([0.6] * (num_classes - 1), k=-1) + np.diag([0.6], k=num_classes - 1) + np.diag([0.0] * num_classes, k=0)
        w = 10.0

        mhp = MHP(mu=m, alpha=a, omega=w)
        num_train = 2000
        num_val = 200
        num_test = 200

        with temporary_seed(13579):
            data_fn = partial(pinwheel, num_classes=num_classes)
            train_set = [generate(mhp, data_fn, ndim=2, num_classes=num_classes) for _ in range(num_train)]
            val_set = [generate(mhp, data_fn, ndim=2, num_classes=num_classes) for _ in range(num_val)]
            test_set = [generate(mhp, data_fn, ndim=2, num_classes=num_classes) for _ in range(num_test)]

        split_set = {
            "train": train_set,
            "val": val_set,
            "test": test_set,
        }

        super().__init__(train_set, split_set[split], split == "split")


class GMMHawkes(SpatioTemporalDataset):

    def __init__(self, split="train"):
        num_classes = 3
        alpha = 0.6
        m = np.array([0.1] * num_classes)
        a = np.diag([alpha] * (num_classes - 1), k=-1) + np.diag([alpha], k=num_classes - 1) + np.diag([0.0] * num_classes, k=0)
        w = 3.0

        mhp = MHP(mu=m, alpha=a, omega=w)
        num_train = 2000
        num_val = 200
        num_test = 200

        with temporary_seed(13579):
            data_fn = gmm
            train_set = [generate(mhp, data_fn, ndim=1, num_classes=3) for _ in range(num_train)]
            val_set = [generate(mhp, data_fn, ndim=1, num_classes=3) for _ in range(num_val)]
            test_set = [generate(mhp, data_fn, ndim=1, num_classes=3) for _ in range(num_test)]

        split_set = {
            "train": train_set,
            "val": val_set,
            "test": test_set,
        }

        super().__init__(train_set, split_set[split], split == "split")


def generate(mhp, data_fn, ndim, num_classes):
    mhp.generate_seq(END_TIME)
    event_times, classes = zip(*mhp.data)
    classes = np.concatenate(classes)
    n = len(event_times)

    data = data_fn(n)
    seq = np.zeros((n, ndim + 1))
    seq[:, 0] = event_times
    for i, data_i in enumerate(np.split(data, num_classes, axis=0)):
        seq[:, 1:] = seq[:, 1:] + data_i * (i == classes)[:, None]
    return seq


def pinwheel(num_samples, num_classes):
    radial_std = 0.3
    tangential_std = 0.1
    num_per_class = num_samples
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = np.random.randn(num_classes * num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return 2 * np.einsum("ti,tij->tj", features, rotations)


def gmm(num_samples):
    m = np.linspace(-2, 2, 3).reshape(3, 1)
    std = 0.2
    return (np.random.randn(1, num_samples) * std + m).reshape(-1, 1)


@contextlib.contextmanager
def temporary_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


if __name__ == "__main__":

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_classes = 10

    rng = np.random.RandomState(13579)
    data = pinwheel(num_classes, 1000, rng)

    for i, data_i in enumerate(np.split(data, num_classes, axis=0)):
        plt.scatter(data_i[:, 0], data_i[:, 1], c=f"C{i}", s=2)

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.savefig(f"pinwheel{num_classes}.png")
