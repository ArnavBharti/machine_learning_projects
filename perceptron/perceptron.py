import numpy as np
from numpy.typing import NDArray 


# Python 3.x onwards classes implicitly inherits from object and it can be omitted.
# Python 2.x requires explicit inheritance from object to have newer features.
class Perceptron(object):
    """Perceptron Classifier

    Parameters
    -----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Epoch

    Attributes
    -----------
    w_: 1d-array
        Weights after fitting
    errors_: list
        Number of misclassifications in every epoch

    """

    def __init__(self, eta: float = 0.1, n_iter: int = 10) -> None:
        self.eta = eta
        self.n_iter = n_iter

    # Returning self because it enables method chaining.
    # self.errors_ isn't neccessary for algorithm.
    # Variable followed by '_' is for variables initialized outside __init__.
    def fit(self, X: NDArray, y: NDArray) -> object:
        """Fits the model to the data

        Parameters
        -----------
        X: {array-like}, shape = [n_samples, n_features]
            Training inputs where n_samples is number of samples and n_features is number of features.
        y: 1d-array, shape = [n_samples]
            Target values

        Returns
        --------
        self: object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X: NDArray) -> NDArray:
        """Calculate net input

        Parameters
        -----------
        X: 1d-array 
            Array of features of a sample

        """
        # Dot product using numpy achieves vectorization
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X: NDArray) -> NDArray:
        """Step function"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

