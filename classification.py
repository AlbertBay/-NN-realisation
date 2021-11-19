import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    EPS = 1e-5

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        y_pred = np.zeros(distances.shape[0])
        if self._weights == 'uniform':
            for i in range(distances.shape[0]):
                y_nearest = self._labels[indices[i]]
                # https://stackoverflow.com/a/28736715
                values, counts = np.unique(y_nearest, return_counts=True)
                y_pred[i] = values[np.argmax(counts)]
            return y_pred
        else:
            for i in range(distances.shape[0]):
                y_nearest = self._labels[indices[i]]
                score = dict.fromkeys(np.unique(y_nearest), 0)
                for j in range(distances.shape[1]):
                    score[y_nearest[j]] += 1/(distances[i, j] + self.EPS)
                    y_pred[i] = max(score, key=score.get)
            return y_pred

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedKNNClassifier(KNNClassifier):
    '''
    Нам нужен этот класс, потому что мы хотим поддержку обработки батчами
    в том числе для классов поиска соседей из sklearn
    '''

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform', batch_size=None):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights,
            metric=metric,
        )
        self._batch_size = batch_size

    def kneighbors(self, X, return_distance=False):
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=return_distance)
        else:
            if return_distance:
                distances = []
                indexes = []
                for i_min in range(0, X.shape[0], self._batch_size):
                    i_max = min(i_min + self._batch_size, X.shape[0])
                    dist, ind = super().kneighbors(X[i_min:i_max], return_distance=return_distance)
                    distances.append(dist)
                    indexes.append(ind)
                return np.vstack(distances), np.vstack(indexes)
            else:
                indexes = []
                for i_min in range(0, X.shape[0], self._batch_size):
                    i_max = min(i_min + self._batch_size, X.shape[0])
                    ind = super().kneighbors(X[i_min:i_max], return_distance=return_distance)
                    indexes.append(ind)
                return np.vstack(indexes)
