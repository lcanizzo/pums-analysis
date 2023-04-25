#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

def main():
    staged = './compiled_data/staged'
    data = pd.read_csv(os.path.join(staged, "all.csv"))
    data.reset_index()
    print('data size: ', data.index.size)

    ## train test split
    hrs_wrkd_x = data.drop(['income_under_20k'], axis=1)
    hrs_wrkd_y = data['income_under_20k']

    x_train, x_test, y_train, y_test = train_test_split(
        hrs_wrkd_x,
        hrs_wrkd_y,
        test_size=0.3,
        stratify=hrs_wrkd_y,
        random_state=42)

    print('\n')
    print(f'x_train size: {x_train.index.size}')
    print(f'y_train size: {y_train.index.size}')
    print(f'x_test size: {x_test.index.size}')
    print(f'y_test size: {y_test.index.size}')

    ## fill nan in x_train data
    print('\n')
    print('\nFill nan in x_train data')
    print('x_train size pre imputation: ', x_train.index.size)
    x_train_imputed = x_train.apply(lambda x: x.fillna(x.value_counts().index[0]))
    x_train = x_train.dropna()
    x_train_imputed = x_train_imputed.dropna()
    print('x_train size post-imputation dropped na: ', x_train_imputed.index.size)
    print('x_train size dropped na: ', x_train.index.size)

    ## fill nan in x_test data
    print('\n')
    print('\nFill nan in x_test data')
    print('x_test size pre imputation: ', x_test.index.size)
    x_test_imputed = x_test.apply(lambda x: x.fillna(x.value_counts().index[0]))
    x_test = x_test.dropna()
    x_test_imputed = x_test_imputed.dropna()
    print('x_test size post-imputation dropped na: ', x_test_imputed.index.size)
    print('x_test size dropped na: ', x_test.index.size)

    x_train_imputed.to_csv(f'./compiled_data/train/x.csv', index=False)
    x_test_imputed.to_csv(f'./compiled_data/test/x.csv', index=False)
    y_train.to_csv(f'./compiled_data/train/y.csv', index=False)
    y_test.to_csv(f'./compiled_data/test/y.csv', index=False)

if __name__ == "__main__":
    from process_timer import time_execution

    time_execution(main)
#%%