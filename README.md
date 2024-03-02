<meta name="google-site-verification" content="Q17v58M2kfbNl_NGiqbFudpkB8_i5R3Sq2AZsRRzDrI" /> <br>
# MLKIT: A Machine Learning Framework for MicroPython
## A Machine Learning framework in MicroPython
***MLKIT*** is a machine learning framework designed for tiny bare-metal controllers who supports micropython. The machine learning code is primarily developed from scratch. The APIs are developed in such a way so that compatibility with popular machine learning tools such as Scikit-Learn is maintained. <br>
Note: Some portion of code is build on Numpy array. You need [ULAB compatible](https://github.com/v923z/micropython-ulab) firmware for Numpy support and to run the code. Some pre-compiled firmware of different port can be found in **firmware** directory.<br>

**Acknowledgement:** Some part of the source code is taken from [here](https://github.com/patrickloeber/MLfromscratch) and modified for running in MicroPython environment.

**Highlights and supported features:** <br>
1. Logistic Regression
2. K-Nearest Neighbors (KNN) classifier
3. Single Layer Perceptron

**Data Processing**
1. csv_read() #Reading CSV file
2. StandardScaler() # Scikit-learn's standard scaler
3. train_test_split() # splitting into train and test set

**Metrics**
1. accuracy score()
2. classification_report()
3. confusion_matrix()

## Usages
Simply download the ***mlkit*** directory and copy it into supported board. You can use editor such as Thonny for editing, running and copying files/folders from PC to board and vice versa. Then run any test code provided (E.g. test_knn01.py). The test script should be outside *mlkit* directory as shown in repo.

## APIs
### mlkit.preprocessing
***StandardScaler()*** <br>
```python
from mlkit.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(X)
scaler = scaler.transform(X)
```
### mlkit.model_selection <br>
***train_test_split()*** <br>
Splitting the dataset into train and test set with specific test size and random state.<br>

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
```
### mlkit.metrics <br>
Note: true_labels and predicted_labels are Numpy array <br>
***accuracy_score()***
```python
from mlkit.metrics import accuracy_score
accuracy = accuracy_score(true_labels, predicted_labels)
```
***classification_report()***
```python
from mlkit.metrics import classification_report
classification_report(true_labels, predicted_labels,num_labels=<number of labels>)
```
***confusion_matrix()*** <br>
Note: True and Predicted labels must be rounded 
```python
from mlkit.metrics import confusion_matrix
confusion_matrix(true_labels, predicted_labels,num_labels=<number of labels>)
```
### mlkit.utils
***csv_read()*** <br>
Reading the content of a comma separated format file from base directory and storing the content into Numpy array. Known issue: Currently support for numbers only, not characters.If the file contain string or character or header string/column, it will throw an error. <br>
```python
from ulab import numpy as np
data = csv_read('foo.csv')
```
***np_round()*** <br>
Rounding 1-D Numpy array upto specified decimal digits
<br>
```python
from ulab import numpy as np
from mlkit.utils import np_round
x = np.array([1.122, 2.664, 9.883, 4.663])
y = np_round(x) #rounding upto zero decimal places, default=0
y1 = np_round(x,2) #rounding upto two decimal places
```
<br>**Note:** rounding function is not implemented in ulab, thats why you need to import it separately from implemented custom functions.<br>
### mlkit.linear_model
***LogisticRegression()*** <br>
Initialize LogisticRegression classifier object. <br>

```python
from ulab import numpy as np
from mlkit.utils import np_round
from mlkit.activation import Sigmoid
from mlkit.linear_model import LogisticRegression
#Generate data
x = np.linspace(-5,5,20).reshape((-1, 1))
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
clf = LogisticRegression(gradient_descent=True)
clf.fit(x,y)
x_test=np.array([-2.0, -3.7, -4.1, -2.8, 0.9, 2.4, 4.2, 5.1]).reshape((-1,1))
ypred = np_round(clf.predict(x_test))
print(ypred)
```
<br>

***KNeighborsClassifier()*** <br>
K-Nearest neighbors classifier. In this version, only Euclidean distance method is used to calculate the distance among different features.<br>
```python
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
```
***Perceptron()*** <br>
A single layer Perceptron with specified activation function.<br>
```python
from mlkit.linear_model import Perceptron
clf = Perceptron(n_iterations=1000,learning_rate=0.001,loss=CrossEntropy,activation_function=Sigmoid)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```
**Note:** Currently supported loss functions are *SquareLoss* and *CrossEntropy* <br>
