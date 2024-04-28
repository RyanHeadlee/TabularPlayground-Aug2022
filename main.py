import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

"""
Code used for the Kaggle contest "Tabular Playground Series August 2022"
Link: https://www.kaggle.com/competitions/tabular-playground-series-aug-2022
Best Model: Logistic Regression with a score of 0.58912
"""

pd.set_option("display.max_columns", None)

# Read the train and test datasets
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
combined = [df_train, df_test]

# Drop id from train set
df_train.drop("id", axis=1, inplace=True)

# print(df_train.info())
# print(df_train.describe())
# print(df_train.describe(include=["O"]))

# Create area feature by multiplying attributes 2 and 3, then drop those attributes
for dataset in combined:
    dataset["area"] = dataset["attribute_2"] * dataset["attribute_3"]
    dataset.drop(["attribute_2", "attribute_3"], axis=1, inplace=True)

# Create new feature "has_null" if index has any missing values = True, otherwise False
for dataset in combined:
    for feature in df_train.drop("failure", axis=1).columns.values:
        dataset["has_null"] = dataset.isnull().any(axis=1)

# Fill any missing data with median value of the feature
for dataset in combined:
    for feature in dataset.columns:
        if dataset[feature].isnull().sum() > 0:
            dataset[feature] = dataset[feature].fillna(dataset[feature].median())

# Create average & stdev features from measurements 3 - 16
for dataset in combined:
    df_measurements = dataset[
        [
            "measurement_3",
            "measurement_4",
            "measurement_5",
            "measurement_6",
            "measurement_7",
            "measurement_8",
            "measurement_9",
            "measurement_10",
            "measurement_11",
            "measurement_12",
            "measurement_13",
            "measurement_14",
            "measurement_15",
            "measurement_16",
        ]
    ]
    dataset["average"] = df_measurements.mean(axis=1)
    dataset["stdev"] = df_measurements.std(axis=1)
    dataset.drop(df_measurements, axis=1, inplace=True)

# Normalize measurements 0 - 2 (square root transform) & loading (log transform)
for dataset in combined:
    for feature in ["measurement_0", "measurement_1", "measurement_2"]:
        dataset[feature] = np.sqrt(dataset[feature])

    dataset["loading"] = np.log(dataset["loading"])

# sns.histplot(df_train["measurement_0"], kde=True)
# sns.histplot(df_train["measurement_1"], kde=True)
# sns.histplot(df_train["measurement_2"], kde=True)
# sns.histplot(df_train["loading"], kde=True)
# plt.show()

# Concat df_train w/o failure and df_test w/o id into a single data frame
df_whole = pd.concat([df_train.drop("failure", axis=1), df_test.drop("id", axis=1)])

# Column transformer to scale numerical features & encode categorical features
ct = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse_output=False), make_column_selector(dtype_include=object)),
)

# Scale values & encode categorical features
df_whole = ct.fit_transform(df_whole)

# Split the data into test and train
train_rows = df_train.shape[0]
test_rows = df_test.shape[0]
x_train = df_whole[:train_rows]
x_test = df_whole[-test_rows:]
y_train = df_train["failure"]

# Create the model (Logistic Regression)
# Kaggle Private Score: 0.58912
model = LogisticRegression()
model.fit(x_train, y_train)
predictions = model.predict_proba(x_test)

# # Create the model (KNeighbors)
# # Kaggle Private Score: 0.51692 (pre has_null)
# model = KNeighborsClassifier()
# model.fit(x_train, y_train)
# predictions = model.predict_proba(x_test)

# # Create the model (Support Vector)
# # Kaggle Private Score: 0.52281
# model = SVC(probability=True)
# model.fit(x_train, y_train)
# predictions = model.predict_proba(x_test)

# # Create the model (Perceptron)
# # Kaggle Private Score: 0.52035 (isotonic)
# # Kaggle Private Score: 0.57652 (sigmoid)
# model = Perceptron()
# clf_isotonic = CalibratedClassifierCV(model, cv=10, method="sigmoid")
# clf_isotonic.fit(x_train, y_train)
# predictions = clf_isotonic.predict_proba(x_test)

# # Best parameters from grid search best_params_ attribute
# best_params = {
#     "criterion": ["log_loss"],
#     "max_depth": [10],
#     "min_samples_split": [10],
#     "min_samples_leaf": [1],
# }

# # Create the model (Decision Tree & GridSearch)
# # Kaggle Private Score: 0.55574
# model = DecisionTreeClassifier()
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=best_params,
#     cv=5,
#     scoring="accuracy",
#     n_jobs=1,
# )
# grid_search.fit(x_train, y_train)
# predictions = grid_search.predict_proba(x_test)

# Create submission df that has id and probability of failure
df_submission = pd.DataFrame(data={"id": df_test["id"], "failure": predictions[:, 1]})

# Convert df to a csv file called "submission.csv"
df_submission.to_csv("submission.csv", index=False)
