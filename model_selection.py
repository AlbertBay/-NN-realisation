from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))

    #  https: // stackoverflow.com / questions / 36063014 / what - does - kfold - in -python - exactly - do
    result = dict.fromkeys(k_list, [])
    knn = BatchedKNNClassifier(n_neighbors=max(k_list), **kwargs)
    for train_index, test_index in cv.split(X):
        knn.fit(X[train_index], y[train_index])
        dist, ind = knn.kneighbors(X[test_index], return_distance=True)
        for k in k_list:
            y_pred = knn._predict_precomputed(ind[:, :k], dist[:, :k])
            result[k] = np.append(result[k], scorer(y[test_index], y_pred))
    return result
