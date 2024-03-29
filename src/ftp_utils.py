"""
Functions & constants to interface with census ftp server.
@author Luca Canizzo
"""
# %%
from io import BytesIO
import re
import zipfile
import sys
sys.path.append("..")

base_path = 'programs-surveys/acs/data/pums/'


def download_file(ftp, path):
    """
    Returns file from census ftp server given file path.
    """
    download = BytesIO()
    try:
        ftp.retrbinary('RETR ' + path, download.write)
        download.seek(0)
        return download
    except Exception as e:
        print(f'\nDownload of path: "{path}" failed with exception: \n{e}')


def extract_csv(ftp, year, type, state):
    """
    Given an ftp instance, survey type ("p" || "h"), and state, 
    extracts the assoc. CSV file in the working dir to the 
    compiled_data/surveys dir.
    """
    file_name = f'csv_{type}{state}.zip'
    zip_file = download_file(ftp, file_name)
    dest = './compiled_data/surveys'

    with zipfile.ZipFile(zip_file, 'r') as zip:
        for info in zip.infolist():
            if re.match(r'.+(\.csv)$', info.filename):
                info.filename = f'{year}_{type}_{state}.csv'
                print(f'extracting "{info.filename}" to {dest}...')
                zip.extract(info, dest)


if __name__ == '__main__':
    from ftplib import FTP

    year = '2019'
    type = 'h'
    state = 'in'

    file_path = 'csv_hin.zip'

    with FTP('ftp2.census.gov') as ftp:
        ftp.login()
        ftp.cwd('programs-surveys/acs/data/pums/2019/1-Year')
        download_file(ftp, file_path)
        # extract_csv(ftp, year, type, state)
# %%
