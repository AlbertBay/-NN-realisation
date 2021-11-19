import numpy as np

from knn.distances import euclidean_distance, cosine_distance


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        # строки - обьекты из теста
        # столбцы - обьекты из трейна
        ranks = self._metric_func(X, self._X)
        print(ranks.shape)
        if return_distance:
            if self.n_neighbors < ranks.shape[1]:
                ind = np.argpartition(ranks, self.n_neighbors-1, axis=1)[:, :self.n_neighbors]
                dist = np.take_along_axis(ranks, ind, axis=1)
                ind = np.take_along_axis(ind, np.argsort(dist), axis=1)
                dist = np.take_along_axis(ranks, ind, axis=1)
            else:
                ind = np.argsort(ranks, axis=1)[:, :self.n_neighbors]
                dist = np.take_along_axis(ranks, ind, axis=1)
            return dist, ind
        else:
            if self.n_neighbors < ranks.shape[1]:
                ind = np.argpartition(ranks, self.n_neighbors - 1, axis=1)[:, :self.n_neighbors]
                dist = np.take_along_axis(ranks, ind, axis=1)
                ind = np.take_along_axis(ind, np.argsort(dist), axis=1)
            else:
                ind = np.argsort(ranks, axis=1)[:, :self.n_neighbors]
            return ind
