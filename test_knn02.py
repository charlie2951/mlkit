#Test function for KNN classifier in Micropython
from ulab import numpy as np
from mlkit.neighbors import KNeighborsClassifier
from mlkit.preprocessing import StandardScaler
from mlkit.model_selection import train_test_split
from mlkit.metrics import classification_report
from mlkit.utils import np_round, csv_read
# Sample data IRIS dataset
data = csv_read('iris.csv')
X=data[:,0:4]
y=data[:,4]
scaler = StandardScaler()
scaler.fit(X)
scaler.transform(X)
#print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=None)
    
# Initialize and fit the classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
     
# Predict
predictions = knn.predict(X_test)
print("Predictions:", predictions)
classification_report(y_test, predictions)

