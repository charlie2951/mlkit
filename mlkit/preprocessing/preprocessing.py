#Script for preprocessing of data
from ulab import numpy as np

class StandardScaler:
    """
    Implements a standard scaler for numerical data.

    Attributes:
        mean_ (ndarray): The mean of each feature in the training data.
        scale_ (ndarray): The standard deviation of each feature in the training data.

    Methods:
        fit(X): Fits the scaler to the training data X.
        transform(X): Transforms the data X using the fitted scaler.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Fits the scaler to the training data X.

        Args:
            X (ndarray): The training data, a 2D array of numerical features.

        Returns:
            StandardScaler: The fitted scaler object.
        """

        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a NumPy array.")

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        # Handle cases where standard deviation is zero
        self.scale_[self.scale_ == 0.0] = 1.0

        return self

    def transform(self, X):
        """
        Transforms the data X using the fitted scaler.

        Args:
            X (ndarray): The data to be transformed, a 2D array of numerical features.

        Returns:
            ndarray: The transformed data.
        """

        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a NumPy array.")

        
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before transforming.")

        return (X - self.mean_) / self.scale_

# Example usage:
#X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#scaler = StandardScaler()
#scaler.fit(X)
#X_scaled = scaler.transform(X)
#print(X_scaled)

