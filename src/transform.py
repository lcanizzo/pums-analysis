"""
Transform raw csv data to staging format.
"""
# %%
import numpy as np
import pandas as pd
import json
import re
from _constants import combinations
from dictionary_utils import get_values_dict, prefix_val, skip_map_columns, \
    custom_transform_columns

error_cols = []
error_tuples = []


def is_empty_value(value):
    """
    Takes a survey response value.
    Returns T/F if it is not a number.
    """
    return pd.isna(value) \
        or pd.isnull(value) \
        or value == 'did not report' \
        or bool(re.match(r'^\s*$', str(value)))


def normalize_null_vals(reported_val):
    """
    Takes a reported value and returns a normalized NaN is null, nan, empty...
    Else returns reported value.
    """
    if is_empty_value(reported_val):
        return np.NaN
    else:
        return reported_val


def set_val_from_dictionary(col, reported_val, dict):
    """
    Given a colun name, value, and data dictionary, returns the human readable
    value for the given column.
    """
    match_val = reported_val

    # normalize return for null and nan
    if is_empty_value(reported_val):
        return np.NaN

    # normalize numeric values to standard number string
    if isinstance(reported_val, float):
        int_val = int(reported_val)
        match_val = str(int_val)
    if isinstance(reported_val, int):
        match_val = str(reported_val)

    match_val = prefix_val(col, match_val)

    # skip matching for skip columns
    if col in skip_map_columns:
        return match_val
    
    # use custom fn for custom columns
    if col in custom_transform_columns:
        return custom_transform_columns[col](match_val)

    # get dictionary value
    if match_val in dict[col]:
        return dict[col][match_val]

    if col not in error_cols:
        print(f'value "{match_val}" for col: "{col}" is missing in the dict.')
        error_cols.append(col)
    if (col, match_val) not in error_tuples:
        error_tuples.append((col, match_val))

    return reported_val


def create_transform_output(combination):
    """
    Given a year, state, and survey type as a list, retrieves the local copy of
    survey data and data dictionary for the given year, transforms data values to
    dictionary equivalent. 
    Output: compiled_data/staged/{year}_{type}_{state}.csv
    """
    [y, state, type] = combination
    year = int(y)
    print(f'\ncreate_transform_output({year}, {state}, {type})')
    data = pd.read_csv(f'./compiled_data/surveys/{year}_{type}_{state}.csv')
    defined_types = pd.read_json(
        f'./compiled_data/types/{str(year)}.json', orient='index')[0]
    data_type = {}

    # drop columns & set data types
    use_cols = pd.read_csv('./configs/col_name_map.csv')
    drop_cols = []

    for col in data:
        if col not in list(use_cols['PUMS_COL_NAME']):
            drop_cols.append(col)
        else:
            pretty_name = use_cols.loc[use_cols['PUMS_COL_NAME'] == col] \
                ['column'].item()

            if pretty_name in defined_types:
                data_type[pretty_name] = defined_types[pretty_name]

    data.drop(drop_cols, axis=1, inplace=True)
    
    # update column names
    with open('./compiled_data/dictionaries/col_name_map.json') as json_file:
        column_name_map = json.load(json_file)
        data = data.rename(column_name_map, axis='columns')

    # trim rows
    if len(data.index) > 5000:
        data = data.tail(5000)

    # transform column values
    try:
        vals_dict = get_values_dict(year)
        for col in data:
            if col in vals_dict:
                data[col] = data[col].map(
                    lambda val: set_val_from_dictionary(col, val, vals_dict)
                )
            elif col in data_type:
                data[col] = data[col].map(
                    lambda val: normalize_null_vals(val)
                )
    except Exception as e:
        print('\n\nEXCEPTION')
        print(e)

    # set data types
    data = data.astype(data_type)

    # log value errors
    if error_tuples:
        error_csv_path = f'val_errors/{year}_{state}_{type}.csv'
        with open(error_csv_path, 'w+') as log_file:
            log_file.write('value,column\n')
            for error_tuple in error_tuples:
                c, v = error_tuple
                log_file.write(f'{v},{c}\n')

    # add year column
    year_series = pd.Series(
        [year] * len(data), index=data.index, name='Year')
    data = pd.concat([data, year_series], axis=1)

    # drop nan class
    data.dropna(
        subset=['AvgHoursWorkedPerWeek'],
        inplace=True)

    # bin hours worked per week
    hr_bins = np.arange(0, 101, 10)
    hr_labels = [
        '0-9',
        '10-19',
        '20-29',
        '30-39',
        '40-49',
        '50-59',
        '60-69',
        '70-79',
        '80-89',
        '90-100+']

    data['weekly_hrs_worked'] = pd.Categorical(
        pd.cut(
            data['AvgHoursWorkedPerWeek'],
            bins=hr_bins,
            labels=hr_labels,
            include_lowest=True
        ),
        categories=hr_labels,
        ordered=True)

    # write to compiled_data/staged directors
    data.to_csv(f'./compiled_data/staged/{year}_{type}_{state}.csv', index=False)


if __name__ == "__main__":
    import glob
    import os
    from multiprocessing import Pool
    from process_timer import time_execution
    from dictionary_utils import create_name_map_json, split_dictionaries

    def process():
        create_name_map_json()
        split_dictionaries()

        # transform responses to dict values
        with Pool() as pool:
            try:
                pool.map(create_transform_output, combinations)
            except Exception as e:
                pool.terminate()
        
        # prepare class label
        staged = './compiled_data/staged'
        files = glob.glob(os.path.join(staged, "*.csv"))
        data = pd.concat((pd.read_csv(f) for f in files))
        data.reset_index()

        data.to_csv(f'./compiled_data/staged/all.csv', index=False)

    time_execution(process)

# %%
