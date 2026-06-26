# Titanic Survival Prediction using Random Forest

## Overview

This project is part of the Kaggle **Titanic - Machine Learning from Disaster** competition. The objective is to predict whether a passenger survived the Titanic disaster using a Random Forest Classifier.

## Dataset

The project uses two datasets provided by Kaggle:

* **train.csv** – Contains passenger information along with the target variable (`Survived`).
* **test.csv** – Contains passenger information without the target variable.

## Features Used

The following features were selected for training the model:

* `Pclass` – Passenger class
* `Sex` – Gender of the passenger
* `SibSp` – Number of siblings/spouses aboard
* `Parch` – Number of parents/children aboard

Since `Sex` is a categorical feature, it is converted into numerical values using **one-hot encoding** with `pd.get_dummies()`.

## Model

The machine learning model used is **Random Forest Classifier** from Scikit-learn.

### Model Parameters

* `n_estimators = 100`
* `max_depth = 5`
* `random_state = 1`

These parameters create a forest of 100 decision trees with a maximum depth of 5 to improve prediction accuracy while reducing overfitting.

## Workflow

1. Load the training and testing datasets.
2. Select the required features.
3. Convert categorical data into numerical format using one-hot encoding.
4. Train the Random Forest model using the training dataset.
5. Predict survival for the test dataset.
6. Save the predictions in a CSV file named **submission.csv**.

## Output

The generated `submission.csv` file contains:

| PassengerId | Survived |
| ----------- | -------- |
| 892         | 0        |
| 893         | 1        |
| ...         | ...      |

This file can be directly submitted to Kaggle for evaluation.

## Libraries Used

* pandas
* scikit-learn

## Code Summary

```python
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=1
)

model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({
    'PassengerId': test_data.PassengerId,
    'Survived': predictions
})

output.to_csv('submission.csv', index=False)
```

## Conclusion

This exercise demonstrates a basic machine learning workflow using the Random Forest algorithm. It covers data preprocessing, feature selection, model training, prediction, and exporting results in the required submission format for Kaggle competitions.
