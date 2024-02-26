#from mlkit.utils import shuffle_data
from ulab import numpy as np
import random

#Train-test split
"""
def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    #Split the data into train and test sets 
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test
 """

#import numpy as np
#Implementation of random.sample() function from scratch
def random_sample(population, k):
    """
    Return a k length list of unique elements chosen from the population sequence.
    
    Parameters:
        population (list or set): A sequence to sample from.
        k (int): The size of the sample to be drawn.

    Returns:
        A list of k unique elements randomly sampled from the population.
    """
    
    if not isinstance(population, (list, set)):
        raise TypeError("Population must be a list or a set.")
    
    n = len(population)
    if not 0 <= k <= n:
        raise ValueError("Sample larger than population or is negative.")
    
    # Use a set to keep track of selected indices to ensure uniqueness
    selected_indices = set()
    sampled_list = []
    
    # Select k unique elements
    while len(selected_indices) < k:
        index = random.randint(0, n - 1)
        if index not in selected_indices:
            selected_indices.add(index)
            sampled_list.append(population[index])
    
    return sampled_list

#Implementing sklearn train_test_split() from scratch
def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the dataset into train and test subsets.

    Parameters:
        X (array-like): The input data.
        y (array-like): The target labels.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Seed for the random number generator.

    Returns:
        X_train (list or array-like): The training input data.
        X_test (list or array-like): The testing input data.
        y_train (list or array-like): The training target labels.
        y_test (list or array-like): The testing target labels.
    """
    # Set random seed if provided
    if random_state:
        random.seed(random_state)
    
    # Calculate number of samples for test set
    num_test_samples = int(len(X) * test_size)
    
    # Generate random indices for the test set
    test_indices = random_sample(list(range(len(X))), num_test_samples)
    
    # Initialize train and test datasets
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    # Split data into train and test sets based on the generated indices
    for i in range(len(X)):
        if i in test_indices:
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

