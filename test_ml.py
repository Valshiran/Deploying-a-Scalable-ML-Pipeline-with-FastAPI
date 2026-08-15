import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model


def test_compute_model_metrics():
    """
    Test that compute_model_metrics calculates precision, recall, and F1 score correctly.
    """
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Validate output types (floats or numeric types)
    assert isinstance(precision, (float, np.floating)), "Precision must be a float."
    assert isinstance(recall, (float, np.floating)), "Recall must be a float."
    assert isinstance(fbeta, (float, np.floating)), "F1-score must be a float."

    # Validate exact known outputs
    # True Positives = 2, False Positives = 1, False Negatives = 1
    # Precision = 2 / (2 + 1) = 0.6667
    # Recall = 2 / (2 + 1) = 0.6667
    assert pytest.approx(precision, 0.001) == 2 / 3
    assert pytest.approx(recall, 0.001) == 2 / 3
    assert pytest.approx(fbeta, 0.001) == 2 / 3


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
