# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

**Developer:** Valshiran
- **Model Date:** August 2026
- **Model Version:** 1.0.0
- **Model Type:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`) trained on tabular demographic data.
- **License:** MIT License.
- **Feedback & Questions:** For issues or inquiries regarding this model, please submit an issue in the project GitHub repository.

## Intended Use

- **Primary Intended Uses:** This model is designed to predict whether an individual's annual income exceeds $50,000 based on census demographic features.
- **Primary Intended Users:** Machine learning engineers, data scientists, and educational reviewers evaluating scalable ML pipeline architectures.
- **Out-of-Scope Uses:** This model is strictly intended for educational and benchmark evaluation purposes. It must not be used for credit scoring, employment screening, loan approval, or any automated decision-making system affecting individuals.

## Training Data

- **Dataset:** UCI Machine Learning Repository Census Income (Adult) Dataset.
- **Data Split:** 80% of the original dataset (26,048 samples) was assigned to the training split using `train_test_split` with a fixed `random_state=42`.
- **Preprocessing:** Categorical attributes (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`) were encoded using a fitted `OneHotEncoder` with `sparse=False` and `handle_unknown="ignore"`. The binary target label (`salary`) was processed using `LabelBinarizer`.

## Evaluation Data

- **Dataset:** Holdout test split comprising 20% of the total dataset (6,513 samples).
- **Preprocessing:** Test features were transformed using the identical `OneHotEncoder` and `LabelBinarizer` instances fitted on the training set to prevent data leakage.

## Metrics

The model evaluation uses three binary classification metrics calculated on the holdout test set for the positive class (`>50K`):
- **Precision:** Measures the proportion of correctly predicted high earners out of all positive predictions.
- **Recall:** Measures the proportion of actual high earners correctly identified by the model.
- **F1-Score:** The harmonic mean of precision and recall, balancing false positives and false negatives.

### Model Performance Metrics
- **Precision:** 0.7419
- **Recall:** 0.6384
- **F1-Score:** 0.6863

## Ethical Considerations

- **Demographic Bias:** The underlying dataset contains sensitive attributes including `race`, `sex`, and `native-country`. Historical societal biases present in the 1994 census data directly influence model outcomes.
- **Sliced Performance Variation:** Sliced evaluation across categorical variables (recorded in `slice_output.txt`) demonstrates measurable variation in precision and recall across demographic subgroups. For instance, predictive performance varies across different gender and racial categories due to class imbalance in the training distribution.

## Caveats and Recommendations

- **Outdated Data:** The data was collected during the 1994 US Census and does not reflect current socioeconomic conditions, wage structures, or inflation rates.
- **Demographic Imbalance:** Certain categorical values have small sample counts in the test set, leading to higher variance in slice-level performance metrics.
- **Recommendation:** Users should consult `slice_output.txt` to review subgroup performance before drawing conclusions. The model should never be deployed in real-world administrative or financial environments without comprehensive bias mitigation and legal audit.