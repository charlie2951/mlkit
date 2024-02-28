#Script for testing classification report and confusion matrix
from mlkit.metrics import confusion_matrix, classification_report
from ulab import numpy as np
y_true = np.array([0,1,1,0,1,1.0,0,1])
print(y_true)
y_pred = np.array([1.0,1.0,1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
print(y_pred)
confusion_matrix(y_true,y_pred,num_classes=2)
classification_report(y_true,y_pred,num_classes=2)
