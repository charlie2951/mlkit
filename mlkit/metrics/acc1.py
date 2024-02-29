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

# Example usage:
y_true = np.array([1, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 1.0, 1])

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy Score:", accuracy)

