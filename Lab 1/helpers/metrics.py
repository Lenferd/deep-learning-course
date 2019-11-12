from sklearn.metrics import classification_report as sklearn_classification_report, confusion_matrix


def classification_report(Actual, y_test):
    predictions = (Actual > .5)[0, :]
    labels = (y_test == 1)[0, :]
    sklearn_classification_report(predictions, labels)