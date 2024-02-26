#Micropython code to implement KNN classifier
from ulab import numpy as np
from mlkit.utils import euclidean_distance
from mlkit.utils import np_bincount, np_argsort

class KNeighborsClassifier():
    """ K Nearest Neighbors classifier.

    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the 
        sample that we wish to predict.
    """
    def __init__(self, n_neighbors=3):
        self.k = n_neighbors
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np_argsort(np.array(distances))[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.argmax(np_bincount(k_nearest_labels))
        return most_common
        
