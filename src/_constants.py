# %%
import numpy as np

# dictionaries prepared for 2013 - 2019
YEARS = [*range(2013, 2020, 1)]

RECENT_YEARS = [*range(2013, 2020, 1)]

STATES = ['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga',
          'hi', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 'me',
          'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm',
          'nv', 'ny', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx',
          'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy']

# type options: 'p' or 'h'
TYPES = ['p']

COMBINATIONS = np.array(np.meshgrid(RECENT_YEARS,STATES,TYPES)).T.reshape(-1,3)

# if defined pulls tail = length from each survey, else uses complete dataset.
MAX_LENGTH = 50

if __name__ == '__main__':
    print('combinations:')
    print(COMBINATIONS)
# %%
