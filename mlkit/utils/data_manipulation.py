#from __future__ import division
#from itertools import combinations_with_replacement
from ulab import numpy as np
import math
import sys
import random
from .numpy_extras import np_random_shuffle, np_atleast_1d, np_expand_dims

#Helper functions for data pre-processiong and post-processing

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
   





