###### Different utility functions used by different class 
from ulab import numpy as np
import math
import sys
import random

######## Function to read CSV file #################
def csv_read(file_name):  # function for reading csv file, return file content as numpy array
    f = open(file_name, 'r')
    w = []
    tmp = []
    for each in f:
        w.append(each)
        # print (each)

    # print(w)
    for i in range(len(w)):
        data = w[i].split(",")
        tmp.append(data)
        # print(data)
    file_data = np.array(([[float(y) for y in x] for x in tmp]))
    #file_data = [[float(y) for y in x] for x in tmp]
    return file_data
    
def normalize(X, axis=-1, order=2):
    #Normalize the dataset X 
    l2 = np_atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np_expand_dims(l2, axis)


def standardize(X):
    #Standardize the dataset X 
    X_std = X
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    for col in range(X.shape[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std
    
def shuffle_data(X, y):
    #Random shuffle of the samples in X and y
    idx = np.arange(X.shape[0])
    np_random_shuffle(idx)
    return X[idx], y[idx]

def make_diagonal(x):
    #Converts a vector into an diagonal matrix 
    m = np.zeros((len(x), len(x)))
    for i in range(len(m[0])):
        m[i, i] = x[i]
    return m
    
       
#Implementation of combinations_with_replacement() from scratch
def combinations_with_replacement(iterable, r):
    # combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        return
    indices = [0] * r
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != n - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (r - i)
        yield tuple(pool[i] for i in indices)  


def divide_on_feature(X, feature_i, threshold):
    #Divide dataset based on if sample value on feature index is larger than the given threshold
        
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return [X_1, X_2]
    
def batch_iterator(X, y=None, batch_size=16):
    #Simple batch generator
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]

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

########### Numpy extra functions those are not present in ULAB ###################

#Implementation of numpy.argsort() function
def np_argsort(arr):
    """
    Returns the indices that would sort an array.

    Parameters:
    arr (array_like): Input array.

    Returns:
    ndarray: Array of indices that sort the input array.
    """
    # Create a list of tuples where each tuple contains the element and its index
    indexed_arr = [(elem, idx) for idx, elem in enumerate(arr)]
    
    # Sort the list of tuples based on the first element of each tuple (the array element)
    sorted_arr = sorted(indexed_arr, key=lambda x: x[0])
    
    # Extract and return the indices from the sorted list
    return [idx for _, idx in sorted_arr]

# Example usage:
if __name__ == "__main__":
    arr = [3, 1, 2, 4]
    sorted_indices = argsort(arr)
    print("Sorted indices:", sorted_indices)


#Function to implement numpy.random.shuffle()
def np_random_shuffle(a):
    """
    Shuffles the elements of an array in-place.

    Args:
        a: The array to shuffle.

    Returns:
        None. The elements of the input array are shuffled in-place.
    """

    for i in range(len(a) - 1, 0, -1):
        # Generate a random index in the range [0, i]
        j = random.randrange(i + 1)

        # Swap the elements at indices i and j
        a[i], a[j] = a[j], a[i]
        
#Function equivalent to numpy.expand_dims()
def np_expand_dims(a, axis=-1):
  """
  Expands the shape of an array by inserting new singleton dimensions at the given axis.

  Args:
    a: The input array.
    axis: The axis along which to insert the new dimensions. Default is -1, which inserts
          at the end.

  Returns:
    A new array with the expanded dimensions.
  """

  a = np.asarray(a)  # Ensure input is a NumPy array
  new_shape = list(a.shape)
  new_shape.insert(axis, 1)  # Insert a new dimension of size 1 at the specified axis
  return a.reshape(tuple(new_shape))  # Reshape the array with the new dimensions

#Implementing Numpy round() funcion from scratch in micropython
def np_round(arr, decimal_digits=0):#input numpy array 1d
    arr = arr.tolist() 	
    rounded_list = [round(x, decimal_digits) for x in arr]
    return np.array(rounded_list)

#Function to implement np.atleast_1d()
def np_atleast_1d(*arrays):
    """
    Converts inputs to arrays with at least one dimension.

    Args:
        *arrays: One or more input arrays or scalars.

    Returns:
        A tuple of arrays, each with at least one dimension.
    """

    result = []
    for array in arrays:
        array = np.array(array)  # Ensure array-like inputs are converted to NumPy arrays
        if array.ndim == 0:
            result.append(array[np.newaxis])  # Add a new axis for scalars
        else:
            result.append(array)  # Preserve higher-dimensional arrays

    return tuple(result)  # Return a tuple to match NumPy's behavior

  
#function implementing numpy.power()
def np_power(base, exponent):
    """
    Raises the elements of `base` to the powers in `exponent` element-wise.

    Args:
        base (array_like): The base array.
        exponent (array_like): The exponent array.

    Returns:
        ndarray: The result of raising the elements of `base` to the powers in `exponent`.

    Raises:
        ValueError: If `base` and `exponent` have different shapes.
    """

    if base.shape != exponent.shape:
        raise ValueError("base and exponent must have the same shape")

    result = np.empty_like(base)
    for i in np.ndindex(base.shape):
        result[i] = base[i] ** exponent[i]

    return result
    
#code to implement numpy.unique() with return count
def np_unique(arr, return_counts=True):
  """
  Implements a custom version of numpy.unique() with return_counts=True functionality.

  Args:
    arr: A NumPy array of values.

  Returns:
    A tuple containing:
      unique: A NumPy array of unique values in the input array.
      counts: A NumPy array of counts for each unique value.
  """

  # Initialize empty lists to store unique values and counts
  unique_values = []
  counts = []

  # Iterate through the array
  for value in arr:
    # Check if the value is already in the unique_values list
    if value not in unique_values:
      unique_values.append(value)
      counts.append(1)
    else:
      # Increment the count for the existing value
      index = unique_values.index(value)
      counts[index] += 1

  # Convert lists to NumPy arrays
  unique = np.array(unique_values)
  counts = np.array(counts)
  if(return_counts==True):
    return unique, counts
  else:
    return unique
    
#Implementation of numpy.bincount()
def np_bincount(arr):
  """
  Counts the occurrences of each value in an array.

  Args:
      arr (array_like): The input array.

  Returns:
      ndarray: The array of counts, where the index corresponds to the value and the value is the count.
  """

  # Handle negative values by adding a constant offset
  min_value = np.min(arr)
  if min_value < 0:
    arr += abs(min_value) + 1

  # Create an empty array to store counts
  counts = np.zeros(int(np.max(arr) + 1))

  # Iterate through the array and increment the corresponding count
  for value in arr:
    counts[int(value)] += 1

  return counts         
