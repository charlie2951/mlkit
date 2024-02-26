from ulab import numpy as np
from .numpy_extras import np_unique, np_expand_dims, np_power
import math
import sys

#Implements python copy() module from scratch 
def _copy(obj):
    """
    Create a shallow copy of the given object.

    Parameters:
        obj: The object to be copied.

    Returns:
        A shallow copy of the object.
    """
    if isinstance(obj, list):
        return obj[:]
    elif isinstance(obj, dict):
        return dict(obj)
    elif isinstance(obj, set):
        return set(obj)
    elif isinstance(obj, tuple):
        return tuple(obj)
    elif isinstance(obj, str):
        return obj[:]
    else:
        raise TypeError("Unsupported type for shallow copy")
        
def calculate_entropy(array):#calculate entropy of given array
  """
  Calculates the entropy of an array.

  Args:
    array: A NumPy array of values.

  Returns:
    The entropy of the array.
  """

  # Check if the array is empty
  if len(array) == 0:
    return 0

  # Count the occurrences of each unique value
  unique, counts = np_unique(array, return_counts=True)

  # Calculate the probabilities of each value
  probabilities = counts / len(array)

  # Calculate the entropy using the formula
  entropy = -np.sum(probabilities * np.log2(probabilities))

  return entropy
  
def mean_squared_error(y_true, y_pred):
    #Returns the mean squared error between y_true and y_pred 
    mse = np.mean(np_power(y_true - y_pred, 2))
    return mse


def calculate_variance(X):
    #Return the variance of the features in dataset X 
    mean = np.ones(X.shape) * X.mean(0)
    n_samples = X.shape[0]
    variance = (1 / n_samples) * np.diag(np.dot((X - mean).T, (X - mean)))
    return variance


def calculate_std_dev(X):
    #Calculate the standard deviations of the features in dataset X 
    std_dev = np.sqrt(calculate_variance(X))
    return std_dev


def euclidean_distance(x1, x2):
    """ Calculates the l2 distance between two vectors """
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2) #issue: float obj not subscriptable
    return np.sqrt(distance)



def calculate_covariance_matrix(X, Y=None):
    #Calculate the covariance matrix for the dataset X 
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * np.dot((X - X.mean(axis=0)).T, (Y - Y.mean(axis=0)))

    return np.array(covariance_matrix, dtype=float)
 
  
