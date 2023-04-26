"""
Common utilities for classification files.
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from feature_selection import get_categorical_cols, encode_class

def get_data():
    x_train = pd.read_csv('./compiled_data/train/x.csv')
    y_train = pd.read_csv('./compiled_data/train/y.csv')
    x_test = pd.read_csv('./compiled_data/test/x.csv')
    y_test = pd.read_csv('./compiled_data/test/y.csv')
    return x_train, x_test, y_train, y_test

def get_data_encoded():
    x_train = pd.read_csv('./compiled_data/train/x.csv')
    y_train = pd.read_csv('./compiled_data/train/y.csv')
    x_test = pd.read_csv('./compiled_data/test/x.csv')
    y_test = pd.read_csv('./compiled_data/test/y.csv')

    # get categorical columns
    cat_cols = get_categorical_cols(x_train)
    
    # encode categorical columns
    x_train_cat = pd.get_dummies(x_train[cat_cols], dtype=int)
    x_test_cat = pd.get_dummies(x_test[cat_cols], dtype=int)

    print('\nPre-column intersection counts:')
    print(f'x_train_cat columns size: {x_train_cat.columns.size}')
    print(f'x_test_cat columns size: {x_test_cat.columns.size}')

    # restrict sets to categories present in training data
    common_cat_cols = [
        c for c in set(x_train_cat.columns).intersection(x_test_cat.columns)
    ]
    x_train_cat = x_train_cat[common_cat_cols]
    x_test_cat = x_test_cat[common_cat_cols]
    
    x_train_cont = x_train.drop(cat_cols, axis=1)
    x_test_cont = x_test.drop(cat_cols, axis=1)
    print('\nPost-column intersection counts:')
    print(f'x_train_cat columns size: {x_train_cat.columns.size}')
    print(f'x_test_cat columns size: {x_test_cat.columns.size}')

    x_train_enc = pd.concat([x_train_cat, x_train_cont], axis=1)
    x_test_enc = pd.concat([x_test_cat, x_test_cont], axis=1)
    y_train_enc = encode_class(y_train)
    y_test_enc = encode_class(y_test)

    if __name__ == "__main__":
        print('\nget encoded')
        print('\nx_train columns: ')
        print(x_train.columns)
        print('\nx_train size: ', x_train.index.size)
        print('\nx_train_encoded columns: ')
        print(x_train_enc.columns)
        print('\nx_train_encoded size: ')
        print(x_train_enc.index.size)

    return x_train_enc, x_test_enc, y_train_enc, y_test_enc

def print_accuracy(y_pred, y_test):
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    print('\nPrediction accuracy: ', accuracy)

def print_confusion_matrix(y_pred, y_test, model_name=''):
    [[tp, fn], [fp, tn]] = \
        confusion_matrix(y_test, y_pred)
    print(f'TP: {tp}')
    print(f'FP: {fp}')
    print(f'TN: {tn}')
    print(f'FN: {fn}')
    print(f'TPR: {round(tp / (tp + fn),5)}')
    print(f'TNR: {round(tn / (tn + fp),5)}')
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    x_train_enc, x_test_enc, y_train_enc, y_test_enc = get_data_encoded()
    x_train, x_test, y_train, y_test = get_data()

    print('\n')
    print('\n')
    print('\nx_train_enc head:')
    print(x_train_enc.head(3))
    print('\nx_train head:')
    print(x_train.head(3))
    print('\n')
    print('\n')
    print('\nx_test_enc head:')
    print(x_test_enc.head(3))
    print('\nx_test head:')
    print(x_test.head(3))
    print('\n')
    print('\n')
    print('\ny_train_enc head:')
    print(y_train_enc[:3])
    print('\ny_train head:')
    print(y_train[:3])
    print('\n')
    print('\n')
    print('\ny_test_enc head:')
    print(y_test_enc[:3])
    print('\ny_test head:')
    print(y_test[:3])


#%%