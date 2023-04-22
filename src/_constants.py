# %%
import numpy as np

# dictionaries prepared for 2013 - 2019
years = [*range(2013, 2020, 1)]
recent_years = [*range(2013, 2020, 1)]
states = ['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga',
          'hi', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 'me',
          'mi', 'mn', 'mo', 'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm',
          'nv', 'ny', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx',
          'ut', 'va', 'vt', 'wa', 'wi', 'wv', 'wy']
# type options: 'p' or 'h'
types = ['p']
combinations = np.array(np.meshgrid(recent_years,states,types)).T.reshape(-1,3)

if __name__ == '__main__':
    print('combinations:')
    print(combinations)
# %%
