'''
A module contains utility functions used throughout this project.

Functions:
    check_valid_csv(): Checks if a csv file exists and contains data.
    check_gpu(): Checks GPU availability for parallel computing.
    get_file_ctime(): Get file creation time as a formatted string.
    get_timestamp(): Get current time as a formatted string.
    timed_tun(): Runs a function with a timer.
    ...
'''

# standard imports
import csv
import datetime
import json
import os
import pickle
import sys
import typing

def check_valid_csv(csv_path: str) -> bool:
    '''Checks if a csv file exists and contains data.'''

    # FALSE if file does not exist
    if not os.path.exists(csv_path):
        return False

    # FALSE if only contains a header (no content from row 1)
    with open(csv_path, mode='r', encoding='utf-8') as file:
        rows = list(csv.reader(file))
        if not rows[1:]:
            return False

    # TRUE otherwise
    return True

def get_file_ctime(filepath: str, t_format: str='%Y%m%d_%H%M%S') -> str:
    '''
    Get file creation time as a string specified by `t_format`.

    Args:
        filepath (str): To the file to be checked.
        t_format (str, optional): Sets time string format
            (default: 20001234_567).
    '''

    # get creation time
    creation_time = os.path.getctime(filepath)
    # format and return
    return datetime.datetime.fromtimestamp(creation_time).strftime(t_format)

def get_timestamp(t_format: str='%Y%m%d_%H%M%S') -> str:
    '''
    Get current time as a string specified by `t_format`.

    Args:
        t_format (str, optional): Sets time string format
            (default: 20001234_567).
    '''

    # return formatted time string
    return datetime.datetime.now().strftime(t_format)

def print_status(lines: list):
    '''Helper to print multiple lines refreshing'''

    print('\n')
    # Calculate the number of lines that should be refreshed
    num_lines_to_clear = len(lines)
    # Move the cursor up by that number of lines
    sys.stdout.write(f'\033[{num_lines_to_clear}F')
    # Move cursor up by len(lines) and clear lines using ANSI escape codes
    sys.stdout.write('\033[F' * len(lines))  # Move cursor up
    for line in lines:
        sys.stdout.write('\033[K')  # Clear the line
        print(line)
    print('\n')

def load_json(json_fpath: str) -> typing.Any:
    '''Helper to load a json config file.'''

    with open(json_fpath, 'r', encoding='UTF-8') as src:
        return json.load(src)

def write_json(json_fpath: str, src_dict: list | dict) -> None:
    '''Helper to write a json config file from a python dict or list.'''

    with open(json_fpath, 'w', encoding='UTF-8') as file:
        json.dump(src_dict, file, indent=4)

def load_pickle(pickle_fpath: str) -> typing.Any:
    '''Helper to load a .pickle file'''

    with open(pickle_fpath, 'rb') as file:
        return pickle.load(file)

def write_pickle(pickle_fpath: str, src_obj: typing.Any) -> None:
    '''Helper to write a json config file from a python dict or list.'''

    with open(pickle_fpath, 'wb') as file:
        pickle.dump(src_obj, file)

def get_fpaths_from_dir(dirpath: str, suffix: str | None=None) -> list:
    '''List all files from a directory with optional suffix filter.'''

    fpaths = []
    for root, _, files in os.walk(dirpath):
        for file in files:
            if not suffix:
                fpaths.append(os.path.join(root, file))
            else:
                if file.endswith(suffix):
                    fpaths.append(os.path.join(root, file))
    return fpaths
