import sklearn.linear_model as linear_model
import sklearn.neighbors as neighbors


class Model:
    """
    Abstraction for the trainable model, exposing generic fit and predict methods.
    """
    def fit(self, X, y):
        """
        Trains a model usind the dataset provided.

        Input:
        X - float np.array, shape is (n_samples, n_features)
        y - boolean np.array, shape is (n_samples, )
        """
        raise NotImplementedError("Should have implemented fit method")

    def predict(self, X):
        """
        Performs inference on a user-provided data using trained model

        Input:
        X - float np.array, shape is (n_samples, n_features)

        Returns:
        boolean np.array, shape is (n_samples, )
        """
        raise Exception("Should have implemented predict method")


class LinearModel(Model):
    """
    Logistic regression model
    Has two hyperparameters: L2 regularization strength and threshold
    """
    @staticmethod
    def generate_model(hyperparams):
        return LinearModel(hyperparams['regularization_strength'], hyperparams['threshold'])

    def __init__(self, regularization_strength, threshold):
        self.threshold = threshold
        self.model = linear_model.LogisticRegression(C=1.0 / regularization_strength)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.decision_function(X) > self.threshold
    
    def score(self, X, y):
        return self.model.score(X, y)

class KNNModel(Model):
    """
    K-nearest neighbors model.
    Has one hyperparameters: number of neighbors
    """
    @staticmethod
    def generate_model(hyperparams):
        return KNNModel(hyperparams['n_neighbors'])

    def __init__(self, n_neighbors):
        self.model = neighbors.KNeighborsClassifier(n_neighbors)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)