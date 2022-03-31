import pandas as pd
from train_model import *
from ml.model import *

df = pd.read_csv("../preprocessed_data.csv")

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
X_test, y_test, _ , _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


def performance_slices(df, feature):
    for feature in df[feature].unique():
        df_temp = df[df[feature] == feature]
        print(f"fixed feature: {feature}")

        model=train_model(X_train, y_train)
        y_pred=inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
        print(precision)
        print(recall)
        print(fbeta)
    print()

performance_slices(df,"workclass")