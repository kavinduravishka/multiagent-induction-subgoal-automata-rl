import os
import shutil
import numpy as np
import json
from colorama import init, Fore, Back, Style


def get_param(param_dict, param_name, default_value=None):
    if param_dict is not None and param_name in param_dict:
        return param_dict[param_name]
    return default_value


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def rm_dir(dir_name):
    if path_exists(dir_name):
        shutil.rmtree(dir_name, ignore_errors=True)


def rm_dirs(dir_list):
    for dir_name in dir_list:
        rm_dir(dir_name)


def rm_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def rm_files(file_list):
    for filename in file_list:
        rm_file(filename)


def is_file_empty(path):
    return os.path.getsize(path) == 0


def path_exists(path):
    return os.path.exists(path)


def sort_by_ord(input_list):
    input_list.sort(key=lambda s: ord(s.lower()))


def randargmax(input_vector):
    return np.random.choice(np.flatnonzero(input_vector == np.max(input_vector)))


def read_json_file(filepath):
    with open(filepath) as f:
        return json.load(f)


def write_json_obj(obj, filepath):
    with open(filepath, 'w') as f:
        json.dump(obj, f)

def diff_arrays(arr1,arr2, explaination_obj = None, explain_func = None):
    explnation = ""

    assert len(arr1) == len(arr2)
    width = 8

    l = len(arr1)

    for i in range(l):
        if arr1[i] != arr2[i]:
            print(Fore.RED + "{:>{}}".format(arr1[i], width) + Style.RESET_ALL, end = "")
            if explain_func and explaination_obj:
                explnation += str(explain_func(explaination_obj[i]))
        else:
            print("{:>{}}".format(arr1[i], width), end = "")
    
    print("")

    for i in range(l):
        if arr1[i] != arr2[i]:
            print(Fore.RED + "{:>{}}".format(arr2[i], width) + Style.RESET_ALL, end = "")
        else:
            print("{:>{}}".format(arr1[i], width), end = "")

    print("\n")
    print(explnation)
    print("\n==========")
    print()

def check_diff_arrays_crucial_at_least_one_element(arr1, arr2, *crucial_elements):
    assert len(arr1) == len(arr2)

    crucial_change = False

    l = len(arr1)

    for i in range(l):
        if arr1[i] != arr2[i] and (arr1[i] in crucial_elements or arr2[i] in crucial_elements):
            crucial_change = True

    return crucial_change

def check_diff_arrays_crucial_only_one_element_arr2(arr1, arr2, *crucial_elements):
    assert len(arr1) == len(arr2)

    crucial_change = False

    l = len(arr1)

    for i in range(l):
        if arr1[i] != arr2[i] and (arr1[i] not in crucial_elements and arr2[i] in crucial_elements):
            crucial_change = True

    return crucial_change


def check_diff_arrays_crucial_both_elements(arr1, arr2, *crucial_elements):
    assert len(arr1) == len(arr2)

    crucial_change = False

    l = len(arr1)

    for i in range(l):
        if arr1[i] != arr2[i] and (arr1[i] in crucial_elements and arr2[i] in crucial_elements):
            crucial_change = True

    return crucial_change
