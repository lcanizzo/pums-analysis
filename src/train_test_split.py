#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

staged = './compiled_data/staged'
data = pd.read_csv(os.path.join(staged, "all.csv"))
data.reset_index()
print('data size: ', data.index.size)

## plot binned hours worked data
plt.hist(data['weekly_hrs_worked'].sort_values(), bins=10)
plt.title('Average hours worked per week')
plt.show()

## train test split
hrs_wrkd_x = data.drop(['AvgHoursWorkedPerWeek', 'weekly_hrs_worked'], axis=1)
hrs_wrkd_y = data['weekly_hrs_worked']

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

x_train.to_csv(f'./compiled_data/train/x.csv', index=False)
x_test.to_csv(f'./compiled_data/test/x.csv', index=False)
y_train.to_csv(f'./compiled_data/train/y.csv', index=False)
y_test.to_csv(f'./compiled_data/test/y.csv', index=False)
#%%