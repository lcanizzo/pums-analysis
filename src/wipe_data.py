# %%
from os import listdir, remove, path

base_path = './compiled_data'
record_dir = f'{base_path}/surveys'
files = [f for f in listdir(
    record_dir
) if f[-4:] != '.txt' and not path.isdir(f'{record_dir}/{f}')]

for file in files:
    remove(f'{record_dir}/{file}')
