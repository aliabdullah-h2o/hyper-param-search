import numpy as np
import sklearn.datasets as datasets


class Dataset:
    """
    Bare-bones dataset class
    X - input data, boolean np.array, shape is (n_samples, n_features)
    y - labels, boolean np.array, shape is (n_samples,)
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y


def generate_toy_dataset(n_samples=400, n_features=10, seed=42):
    """
    Generates a toy Dataset for binary classification with reasonable defaults.

    Input:
    n_samples - number of samples to generate
    n_features - dimensionality of every sample
    seed - random seed
    """
    X, y = datasets.make_classification(n_samples, n_features, n_redundant=0,
                                        n_clusters_per_class=2, weights=[0.6, 0.4], flip_y=0.1,
                                        random_state=seed)
    return Dataset(X, y.astype(np.bool))
