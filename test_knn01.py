#Test function for KNN classifier in Micropython
from ulab import numpy as np
from mlkit.neighbors import KNeighborsClassifier

# Sample data
X_train = np.array([[1,2,1], [2,3,2], [3,4,2], [4,5,5]])
y_train = np.array([0, 0, 1, 1])
    
# Initialize and fit the classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
    
# Test data
X_test = np.array([[1,2,.4], [4,5,4]])
    
# Predict
predictions = knn.predict(X_test)
print("Predictions:", predictions)


