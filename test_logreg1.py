#Test script for Logistic Regression in MicroPython
from ulab import numpy as np
# Import helper functions
from mlkit.utils import np_round
from mlkit.activation import Sigmoid
from mlkit.linear_model import LogisticRegression

#Generate test data
x = np.linspace(-5,5,20).reshape((-1, 1))
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#print(x)
#print(y)
clf = LogisticRegression(gradient_descent=True)
clf.fit(x,y)
x_test=np.array([-2.0, -3.7, -4.1, -2.8, 0.9, 2.4, 4.2, 5.1]).reshape((-1,1))
ypred = np_round(clf.predict(x_test))
print(ypred)
print("Predicted values are:",ypred)


