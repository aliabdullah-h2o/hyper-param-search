import unittest

from dataset import generate_toy_dataset
from cross_validation import cross_validate
from models import LinearModel, KNNModel


class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        self.toy = generate_toy_dataset()

    def test_linear(self):
        hyperparam_values = {
            'regularization_strength': [0.3, 0.5, 1.0, 3.0],
            'threshold': [0.1, 0.5, 0.9]
        }
        best_model, best_params = cross_validate(self.toy, LinearModel.generate_model, hyperparam_values, 
                                                    5
                                                 # your own arguments here
                                                 )

        self.assertIn(best_params['regularization_strength'], hyperparam_values['regularization_strength'])
        self.assertIn(best_params['threshold'], hyperparam_values['threshold'])
        self.assertIsInstance(best_model, LinearModel)

    def test_knn(self):
        hyperparam_values = {
            'n_neighbors': [1, 3, 30, 60],
        }
        best_model, best_params = cross_validate(self.toy, KNNModel.generate_model, hyperparam_values, 5
                                                 # your own arguments here
                                                 )

        # Toy dataset favors larger neighbor numbers
        self.assertIn(best_params['n_neighbors'], hyperparam_values['n_neighbors'])
        self.assertIsInstance(best_model, KNNModel)


if __name__ == '__main__':
    unittest.main()
