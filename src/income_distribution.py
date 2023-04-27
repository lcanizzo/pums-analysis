#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('./compiled_data/staged/all_class_value.csv')

# set negative income to zero
df['UnadjustedTotalPersonIncome'] = np.where(
    df['UnadjustedTotalPersonIncome'] < 0, 
    0, 
    df['UnadjustedTotalPersonIncome']
)

# bin income
income_bins = np.linspace(0, 185000, 15).round()
income_labels = []

for i in range(0, len(income_bins) - 1):
    start = f'${int(income_bins[i]):,}'
    end = None
    if i < len(income_bins) - 1:
        end = f'${int(income_bins[i+1]):,}'
    if i == len(income_bins) - 2:
        end = end + '+'
    income_labels.append(f'{start} - {end}')

df['personal_income'] = pd.Categorical(
    pd.cut(
        df['UnadjustedTotalPersonIncome'],
        bins=income_bins,
        labels=income_labels,
        include_lowest=True
    ),
    categories=income_labels,
    ordered=True
)

## plot binned income
print('\n')
print('\n')
df['personal_income'].value_counts().plot(kind='bar')
plt.title('Income Range Frequencies')
plt.xlabel('Income')
plt.ylabel('Count')
plt.show()
#%%