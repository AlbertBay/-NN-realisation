import numpy as np


def euclidean_distance(x, y):
    # https://stackoverflow.com/questions/1871536/minimum-euclidean-distance-between-points-in-two-different-numpy-arrays-not-wit/43359192#43359192
    p = np.add.outer(np.sum(x ** 2, axis=1), np.sum(y ** 2, axis=1))
    n = np.dot(x, y.T)
    dists = np.sqrt(p - 2 * n)
    return dists


def cosine_distance(x, y):
    # https://coderedirect.com/questions/305814/find-minimum-cosine-distance-between-two-matrices
    dots = np.dot(x, y.T)
    l2norms = np.sqrt(((x ** 2).sum(1)[:, None]) * ((y ** 2).sum(1)))
    cosine_dists = 1 - (dots / l2norms)
    return cosine_dists
