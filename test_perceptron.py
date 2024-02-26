from ulab import numpy as np

# Import helper functions
from mlkit.utils import np_round, csv_read
from mlkit.metrics import accuracy_score, classification_report
from mlkit.activation import Sigmoid, Relu
from mlkit.loss import CrossEntropy 
from mlkit.linear_model import Perceptron
from mlkit.preprocessing import StandardScaler
from mlkit.model_selection import train_test_split

def main():
    data = csv_read('iris.csv')
    X=data[0:100,0:4]
    y=data[0:100,4].reshape((-1,1))
    #print(ytrain)
    scaler = StandardScaler()
    scaler.fit(X)
    scaler.transform(X)
    #print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=None)
    

    # Perceptron
    clf = Perceptron(n_iterations=5000,
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
    
    classification_report(y_test.reshape((1,-1))[0], np_round(y_pred[0]))


if __name__ == "__main__":
    main()
