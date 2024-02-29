#Logistic Regression class ported in MicroPython
from ulab import numpy as np
import math
from mlkit.utils import make_diagonal
from mlkit.utils import np_round
from mlkit.activation import Sigmoid, Relu, SoftPlus, LeakyRelu, Tanh, Elu
from mlkit.loss import CrossEntropy, SquareLoss

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
        
######################Model for perceptron############################
##--------------------------------------------------------------------##


class Perceptron():
    """The Perceptron. One layer neural network classifier.

    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    activation_function: class
        The activation that shall be used for each neuron.
        Possible choices: Sigmoid, ExpLU, ReLU, LeakyReLU, SoftPlus, TanH
    loss: class
        The loss function used to assess the model's performance.
        Possible choices: SquareLoss, CrossEntropy
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations=20000, activation_function=Sigmoid, loss=SquareLoss, learning_rate=0.001):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.loss = loss()
        self.activation_func = activation_function()
        #self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        _, n_outputs = y.shape

        # Initialize weights between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        rng = np.random.Generator(123)#random seed
        self.W = rng.uniform(-limit, limit, (n_features, n_outputs))
        self.w0 = np.zeros((1, n_outputs))

        for i in range(self.n_iterations):
            # Calculate outputs
            linear_output = np.dot(X, self.W) + self.w0
            y_pred = self.activation_func(linear_output)
            # Calculate the loss gradient w.r.t the input of the activation function
            error_gradient = self.loss.gradient(y, y_pred) * self.activation_func.gradient(linear_output)
            # Calculate the gradient of the loss with respect to each weight
            grad_wrt_w = np.dot(X.T, error_gradient)
            grad_wrt_w0 = np.sum(error_gradient, axis=0) #keepdims ignored
            # Update weights
            self.W  -= self.learning_rate * grad_wrt_w
            self.w0 -= self.learning_rate  * grad_wrt_w0

    # Use the trained model to predict labels of X
    def predict(self, X):
        y_pred = self.activation_func(np.dot(X, self.W) + self.w0)
        return y_pred
        
