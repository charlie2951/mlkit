from ulab import numpy as np

def accuracy_score(y_true, y_pred):
    #Compare y_true to y_pred and return the accuracy 
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


#Classification report
def print_matrix(M, decimals=3):
    """
    Print a matrix one row at a time
        :param M: The matrix to be printed
    """
    for row in M:
        #print("\n")
        print([round(x, decimals) + 0 for x in row])

    
def classification_report(ytrue, ypred):  # print prediction results in terms of metrics and confusion matrix
    tmp = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(ytrue)):
        if ytrue[i] == ypred[i]:  # For accuracy calculation
            tmp += 1
        ##True positive and negative count
        if ytrue[i] == 1 and ypred[i] == 1:  # find true positive
            TP += 1
        if ytrue[i] == 0 and ypred[i] == 0:  # find true negative
            TN += 1
        if ytrue[i] == 0 and ypred[i] == 1:  # find false positive
            FP += 1
        if ytrue[i] == 1 and ypred[i] == 0:  # find false negative
            FN += 1
    accuracy = round((tmp / len(ytrue)),2)
    conf_matrix = [[TP, FP], [FN, TN]]
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    #print(TP, FP, FN, TN)
    print("<-------------------Printing Classification Report----------------->")
    print("\tPrecision: ",round(precision,2))
    print("\tRecall: ",round(recall,2))
    print("\tAccuracy score: " + str(accuracy))
    print("\tF1 score:",round(f1,2))
    print("\nConfusion Matrix:")
    print(print_matrix(conf_matrix))
    
   
