from ulab import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy score of a classification model.

    Args:
    - y_true (list or array-like): The true labels.
    - y_pred (list or array-like): The predicted labels.

    Returns:
    - float: The accuracy score.
    """
    # Ensure the lengths of true labels and predicted labels are the same
    if len(y_true) != len(y_pred):
        raise ValueError("The lengths of true labels and predicted labels must be the same.")

    # Calculate the number of correct predictions
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    # Calculate the total number of predictions
    total_predictions = len(y_true)

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions

    return accuracy

def confusion_matrix(true_labels, predicted_labels, num_classes):
    # Initialize confusion matrix
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    #convert array to list
    true_labels = true_labels.tolist()
    predicted_labels = predicted_labels.tolist()
    # Update confusion matrix
    for true, pred in zip(true_labels, predicted_labels):
        matrix[int(true)][int(pred)] += 1
    
    print("\n###----------Confusion Matrix-------------###")
    
    for row in matrix:
        print(row)
    return np.array(matrix)

def classification_report(true_labels, predicted_labels, num_classes):
    # Initialize variables to store metrics
    precision = [0] * num_classes
    recall = [0] * num_classes
    f1_score = [0] * num_classes
    
    # Calculate true positives, false positives, and false negatives for each class
    for i in range(num_classes):
        true_positive = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == i and pred == i)
        false_positive = sum(1 for true, pred in zip(true_labels, predicted_labels) if true != i and pred == i)
        false_negative = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == i and pred != i)
        
        # Calculate precision
        precision[i] = true_positive / (true_positive + false_positive) if true_positive + false_positive != 0 else 0
        
        # Calculate recall
        recall[i] = true_positive / (true_positive + false_negative) if true_positive + false_negative != 0 else 0
        
        # Calculate F1 score
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if precision[i] + recall[i] != 0 else 0
    
    
    #return precision, recall, f1_score
    # Print metrics
    print("<--------------Printing Classsification Report----------------->")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    
    
   
