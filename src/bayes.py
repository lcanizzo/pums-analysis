#%%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, chi2
from classification_utils import print_accuracy, print_confusion_matrix, \
get_categorical_cols, get_continuous_cols
from process_timer import time_execution

def main():
    np.random.seed(0)

    # Train test split
    df = pd.read_csv('./compiled_data/staged/all.csv')
    
    features_x = df.drop(['income_under_20k'], axis=1)
    class_y = df['income_under_20k']
    X_train, X_test, y_train, y_test = train_test_split(
        features_x,
        class_y,
        test_size=0.2,
        random_state=0
    )

    # Prepare imputer, encoder, and categorical feature selection
    numeric_features = get_continuous_cols(df)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_features = get_categorical_cols(df)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0
    )

    # Gaussian Naive Bayes
    gaussian = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            ("classifier", GaussianNB())
        ]
    )

    gaussian.fit(X_train, y_train)
    y_pred = gaussian.predict(X_test)

    print_accuracy(y_pred, y_test)
    print_confusion_matrix(y_pred, y_test, "Gaussian Naive Bayes")

    # Multinomial Naive Bayes
    multinomial = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            ("classifier", MultinomialNB())
        ]
    )

    y_pred = multinomial.fit(X_train, y_train).predict(X_test)

    print_accuracy(y_pred, y_test)
    print_confusion_matrix(y_pred, y_test, "Multinomial Naive Bayes")

    # Bernoulli Naive Bayes
    bernoulli = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            ("classifier", BernoulliNB())
        ]
    )

    y_pred = bernoulli.fit(X_train, y_train).predict(X_test)

    print_accuracy(y_pred, y_test)
    print_confusion_matrix(y_pred, y_test, "Bernoulli Naive Bayes")

if __name__ == '__main__':
    time_execution(main)
#%%