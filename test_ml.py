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


def test_train_model_algorithm_and_type():
    """
    Test that train_model returns a fitted Scikit-Learn RandomForestClassifier instance.
    """
    X_dummy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y_dummy = np.array([0, 0, 1, 1])

    model = train_model(X_dummy, y_dummy)

    # Verify instance type and fitted state
    assert isinstance(
        model, RandomForestClassifier
    ), "Trained model must be an instance of RandomForestClassifier."
    assert hasattr(model, "classes_"), "Trained model must be fitted with classes_ attribute."
    assert len(model.classes_) == 2, "Binary model must have 2 classes."


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
