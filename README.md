# MLKIT: A Machine Learning Framework for MicroPython
## A Machine Learning framework in MicroPython
***MLKIT*** is a machine learning framework designed for tiny bare-metal controllers who supports micropython. The machine learning code is primarily developed from scratch. The APIs are developed in such a way so that compatibility with popular machine learning tools such as Scikit-Learn is maintained. <br>
Note: Some portion of code is build on Numpy array. You need [ULAB compatible](https://github.com/v923z/micropython-ulab) firmware for Numpy support and to run the code. Some pre-compiled firmware of different port can be found in **firmware** directory.<br>

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
Simply download the ***mlkit*** directory and copy it into supported board. You can use editor such as Thonny for editing, running and copying files/folders from PC to board and vice versa. The run any test code provided (E.g. test_knn01.py). The test script should be outside *mlkit* directory.
