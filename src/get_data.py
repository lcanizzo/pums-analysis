"""
Functions to load historical data from census PUMS
@author Luca Canizzo
"""
#%%
from multiprocessing import Pool
from _constants import RECENT_YEARS, TYPES, COMBINATIONS
from ftplib import FTP
from ftp_utils import base_path, extract_csv

def extract_file(combination):
    """
    Given a state, year, and survey type as a list:
    Connects to the census FTP server and extracts 1-year PUMS CSV
    """
    [y, state, type] = combination
    year = int(y)
    print(f'--- {type} extraction for {state} in {year} started.')
    with FTP('ftp2.census.gov') as ftp:
        ftp.login()
        # cwd to year
        ftp.cwd(base_path + f'{year}')

        # Check if in 1-Year or default dir 
        hasOneYearDir = False
        def setHasDir (retr_line):
            nonlocal hasOneYearDir
            hasOneYearDir = retr_line.endswith('1-Year')
        ftp.retrlines('LIST *1-Year*', setHasDir)

        # Update relative path appropriately
        if hasOneYearDir:
            ftp.cwd('1-Year')

        extract_csv(ftp, year, type, state)



def main():
    print(f'years: {RECENT_YEARS}')
    print(f'types: {TYPES}')
    with Pool() as pool:
        try:
            pool.map(extract_file, COMBINATIONS)
        except Exception as e:
            pool.terminate()

if __name__ == "__main__":
    from process_timer import time_execution

    time_execution(main)    
   
# %%