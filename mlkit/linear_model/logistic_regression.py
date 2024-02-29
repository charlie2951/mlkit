#Logistic Regression class ported in MicroPython
from ulab import numpy as np
import math
from mlkit.utils.data_manipulation import make_diagonal
from mlkit.utils import np_round
from mlkit.activation import Sigmoid


class LogisticRegression():
    """ Logistic Regression classifier.
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If
        false then we use batch optimization by least squares.
    """
    def __init__(self, learning_rate=0.0001, gradient_descent=True):
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        #n_features = np.shape(X)[1]
        n_features = X.shape[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        rng = np.random.Generator(11)
        
        limit = 1 / math.sqrt(n_features)
        self.param = rng.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, n_iterations=10000):
        self._initialize_parameters(X)
        # Tune parameters for n iterations
        for i in range(n_iterations):
            # Make a new prediction
            y_pred = self.sigmoid(np.dot(X,self.param))
            
            if self.gradient_descent:
                # Move against the gradient of the loss function with
                # respect to the parameters to minimize the loss
                self.param -= self.learning_rate * -np.dot((y - y_pred), X)
            else:
                # Make a diagonal matrix of the sigmoid gradient column vector
                diag_gradient = make_diagonal(self.sigmoid.gradient(X.dot(self.param)))
                # Batch opt:
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X):
        #y_pred = np_round(self.sigmoid(np.dot(X,(self.param))))
        y_pred = self.sigmoid(np.dot(X, self.param))
        return np.array(y_pred)
