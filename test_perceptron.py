from ulab import numpy as np

# Import helper functions
from mlkit.utils import np_round, csv_read
from mlkit.metrics import accuracy_score, classification_report
from mlkit.activation import Sigmoid, Relu
from mlkit.loss import CrossEntropy 
from mlkit.linear_model import Perceptron
from mlkit.preprocessing import StandardScaler
from mlkit.model_selection import train_test_split
import time
t1=time.time()
def main():
    data = csv_read('iris.csv')
    X=data[0:100,0:4]
    y=data[0:100,4].reshape((-1,1))
    #print(ytrain)
    scaler = StandardScaler()
    scaler=scaler.fit(X)
    X1=scaler.transform(X)
    print(X1)
    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=5)
    #Sprint(X_test)
    

    # Perceptron
    clf = Perceptron(n_iterations=1000,
        learning_rate=0.001, 
        loss=CrossEntropy,
        activation_function=Sigmoid)
    clf.fit(X_train, y_train)
    #Test data
    #xtest=data[0:100,0:4]
    #ytest=data[0:100,4].reshape((-1,1))
    #x_test=np.array([-2.0, -3.7, -4.1, -2.8, 0.9, 2.4, 4.2, 5.1]).reshape((-1,1))
    #y_test = np.array([0, 0, 0, 0, 1, 1, 1, 1]).reshape((-1,1))
    y_pred = clf.predict(X_test)
    #y_test = np.argmax(y_test, axis=1)
    y_pred = y_pred.reshape((1,-1))
    print(y_pred)
    classification_report(y_test.reshape((1,-1))[0], np_round(y_pred[0]), num_classes=2)
    #confusion_matrix(y_test.reshape((1,-1))[0], np_round(y_pred[0]), num_classes=2)
    print("Time elapsed: "+str(time.time()-t1)+"sec")


if __name__ == "__main__":
    main()
