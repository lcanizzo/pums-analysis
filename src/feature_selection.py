"""
Run feature evaluation on categorical features, and correlation analysis on nontinuous features
"""
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def get_categorical_cols(df):
   """
   Given a dataframe, returns a list of columns that are categorical.
   """
   return df.select_dtypes('object').columns

def get_continuous_cols(df):
   """
   Given a dataframe, returns a list of columns that are continuous.
   """
   categorical_columns = get_categorical_cols(df)
   columns = df.columns
   return [col for col in columns if col not in categorical_columns]

def encode_features_categorical(df):
    """
    Encodes categorical variables and drops continuous variables.
    """
    le = LabelEncoder()
    df_encoded = pd.DataFrame()
    categorical_columns = get_categorical_cols(df)

    for feature in categorical_columns:
        df_encoded[feature] = le.fit_transform(df[feature])
    
    return df_encoded
 
def encode_class(df):
    """
    Encodes class variable.
    """
    le = LabelEncoder()
    le.fit(df.values.ravel())
    df_encoded = le.transform(df.values.ravel())
    return df_encoded
 
def chi2_selection_all(x_train, y_train):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(x_train, y_train.ravel())
    return fs

def get_selected_features(x_train, y_train):
    selected_features = []
    # Chi2 Categorical analysis
    cat_cols = get_categorical_cols(x_train)
    x_train_enc = encode_features_categorical(x_train)
    y_train_enc = encode_class(y_train)
    fs = chi2_selection_all(x_train_enc, y_train_enc)

    # Sort categorical feature scores
    selected_categorical = []
    for i in range(len(fs.scores_)):
        selected_categorical.append({'column': cat_cols[i],'score':fs.scores_[i]})

    selected_categorical.sort(key=lambda x: x['score'], reverse=True)

    # Ranking categorical features
    n = 10

    # Top categorical features
    top_n_categorical = selected_categorical[:n+1]

    if __name__ == "__main__":
        print('\n')
        print(f'\nTop {n} categorical features:\n')

    for i, obj in enumerate(top_n_categorical):
        selected_features.append(obj['column'])

        if __name__ == "__main__":
            print(f'{i+1}). {obj["column"]}: {obj["score"]}')

    # Bottom categorical features
    bottom_n_categorical = list(reversed(selected_categorical))[:n+1]
    
    if __name__ == "__main__":
        print('\n')
        print(f'\nBottom {n} categorical features:\n')

        for i, obj in enumerate(bottom_n_categorical):
            print(f'{i+1}). {obj["column"]}: {obj["score"]}')

    # Continuous feature analysis
    all_data = pd.read_csv('./compiled_data/staged/all.csv')
    continuous_features = get_continuous_cols(all_data)
    continuous_data = all_data[continuous_features]
    cont_corr = continuous_data.corr()
    
    if __name__ == "__main__":
        print('\n')
        print('\nContinous variable correlation coefficients to AvgHoursWorkedPerWeek:')
        sns.heatmap(cont_corr, vmin=-1, vmax=1, annot=True)
        plt.title('Continuous data correlation', pad=12)
        plt.show()

    ranked_cont_corr = cont_corr['AvgHoursWorkedPerWeek'].abs().sort_values(ascending=False)
    ranked_cont_corr.drop('AvgHoursWorkedPerWeek', inplace=True)

    if __name__ == "__main__":
        print('\nAverage hours worked per week continuous correlations (ranked):')
    
    for i in range(0, len(ranked_cont_corr)):
        if ranked_cont_corr.iloc[i] >= 0.3:
            selected_features.append(ranked_cont_corr.index[i])

        if __name__ == "__main__":
            print(f'{i+1}). {ranked_cont_corr.index[i]}: {ranked_cont_corr.iloc[i]}')
    
    if __name__ == "__main__":
        print(f'\nSelected features ({len(selected_features)}):')
        for col in selected_features:
            print(f'    {col}')
    
    return selected_features

if __name__ == "__main__":
    train_path = './compiled_data/train'
    test_path = './compiled_data/test'
    # Load test and train
    x_train = pd.read_csv(f'{train_path}/x.csv')
    x_test = pd.read_csv(f'{test_path}/x.csv')
    y_train = pd.read_csv(f'{train_path}/y.csv')
    y_test = pd.read_csv(f'{test_path}/y.csv')

    selected_features = get_selected_features(x_train, y_train)

    x_train_selected = x_train[selected_features]
    x_test_selected = x_test[selected_features]

    print('\n')
    print('\nx_train size: ', x_train.index.size)
    print('\nx_train columns:')
    print(x_train.columns)

    print('\nx_train_selected size: ', x_train_selected.index.size)
    print('\nx_train_selected columns:')
    print(x_train_selected.columns)

    x_train_selected.to_csv(f'{train_path}/x.csv', index=False)
    x_test_selected.to_csv(f'{test_path}/x.csv', index=False)

    # Note: Ordinal selected features:
    # ordinal_selected = ['TimeOfArrivalAtWork', 'TimeOfDepartureForWork', 'GradeLevelAttending']
    # ordinal_dict_col = ['JWAP', 'JWDP', 'SCHG']


#%%