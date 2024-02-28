from ulab import numpy as np

# Import helper functions
from mlkit.utils import np_round, csv_read
from mlkit.metrics import accuracy_score, classification_report, confusion_matrix
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
    #print(X1)
    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=5)
    #Sprint(X_test)
    

    # Perceptron
    clf = Perceptron(n_iterations=1000,
        learning_rate=0.001, 
        loss=CrossEntropy,
        activation_function=Sigmoid)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #Reshaping actual and predicted values for classification report and confusion matrix
    y_test = y_test.reshape((1,-1))[0]
    y_pred = y_pred.reshape((1,-1))[0]
    #Calculate metrics
    confusion_matrix(y_test, np_round(y_pred), num_classes=2)
    classification_report(y_test, np_round(y_pred), num_classes=2)
    print("Accuracy_score: ",accuracy_score(y_test, np_round(y_pred)))
    
    print("Time elapsed: "+str(time.time()-t1)+"sec")


if __name__ == "__main__":
    main()
