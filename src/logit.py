"""
Runs a logistic regression model with the newton-cholesky solver to
classify test data on whether persons make more than 40k a year.
"""

#%%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, chi2
from classification_utils import print_metrics, \
get_categorical_cols, get_continuous_cols
from process_timer import time_execution

def main():
    np.random.seed(0)

    # Train test split
    df = pd.read_csv('./compiled_data/staged/all_transformed.csv')
    
    features_x = df.drop(['income_over_40k'], axis=1)
    class_y = df['income_over_40k']
    X_train, X_test, y_train, y_test = train_test_split(
        features_x,
        class_y,
        test_size=0.3,
        random_state=0
    )

    # Prepare imputer, scaler / encoder, and categorical feature selection
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=25)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, get_continuous_cols(df)),
            ("cat", categorical_transformer, get_categorical_cols(df)),
        ],
        sparse_threshold=0
    )

    # Logistic Regression
    logit = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            ("classifier", LogisticRegression(solver='newton-cholesky', n_jobs=-1))
        ]
    )

    y_pred = logit.fit(X_train, y_train).predict(X_test)

    print_metrics(y_pred, y_test, "Logit")

if __name__ == '__main__':
    time_execution(main)
#%%