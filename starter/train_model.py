# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import pickle
import dvc

from ml.data import process_data
from ml.model import *

data=pd.read_csv("../preprocessed_data.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

X_test, y_test, _ , _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.

model=train_model(X_train, y_train)

y_pred=inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

print(precision, recall, fbeta)

pickle.dump(model, open("model.pkl","wb"))