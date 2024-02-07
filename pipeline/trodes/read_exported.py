#!/usr/bin/env python3
"""
"""
import os
import warnings
import re
from collections import defaultdict
import pathlib
import numpy as np


def parse_fields(field_str):
    """
    Parses a string of fields into a numpy data type object.

    The input string should be formatted as '<fieldname num*type>' or
    '<fieldname type>'. This function parses the string, extracts the field
    names and data types, and creates a numpy data type object which can be
    used to read data from a binary file.

    Args:
        field_str (str): The string specifying the fields.

    Returns:
        np.dtype: A numpy data type object that describes the structure of
                  the data.

    Raises:
        SystemExit: If the provided field type is not a valid numpy data type.
    """

    # Clean up the string and split it into components
    components = re.split('\s', re.sub(r"\>\<|\>|\<", ' ', field_str).strip())

    dtype_spec = []  # Will hold tuples to specify the numpy data type

    # Iterate over pairs of components (field name and type)
    for i in range(0, len(components), 2):
        field_name = components[i]

        # Default values
        repeat_count = 1
        field_type_str = 'uint32'

        # If the field type string contains a '*', it indicates a repeat count
        if '*' in components[i + 1]:
            split_types = re.split('\*', components[i + 1])
            # Handle both 'num*type' and 'type*num'
            field_type_str = split_types[split_types[0].isdigit()]
            repeat_count = int(split_types[split_types[1].isdigit()])
        else:
            field_type_str = components[i + 1]

        # Convert the field type string to an actual numpy data type
        try:
            field_type = getattr(np, field_type_str)
        except AttributeError:
            print(f"{field_type_str} is not a valid field type.")
            exit(1)
        else:
            dtype_spec.append((str(field_name), field_type, repeat_count))

    return np.dtype(dtype_spec)


def read_trodes_extracted_data_file(filename):
    """
    Reads the content of a Trodes extracted data file.

    This function opens a Trodes file, reads the settings, parses them into a dictionary,
    and then reads the remaining data in the file as a numpy array according to the
    data types specified in the settings. If the settings block does not start correctly,
    it raises an Exception.

    Args:
        filename (str): The path to the Trodes file to be read.

    Returns:
        dict: A dictionary where keys are settings field names and values are the
              corresponding setting values. The actual data from the file is stored
              under the 'data' key as a numpy array.

    Raises:
        Exception: If the settings block in the file does not start with '<Start settings>'.
    """
    print("in read func with " + filename)
    with open(filename, 'rb') as f:
        # The first line of the file should start the settings block
        if f.readline().decode('ascii').strip() != '<Start settings>':
            raise Exception("Settings format not supported")

        # Flag indicating we're reading the settings block
        fields = True
        # Dictionary to hold the settings fields and values
        fields_text = {}

        # Iterate over the lines in the file
        for line in f:
            # If we're still reading the settings block
            if fields:
                line = line.decode('ascii').strip()
                # If we've not reached the end of the settings block, continue reading fields
                if line != '<End settings>':
                    key, value = line.split(': ')
                    fields_text.update({key.lower(): value})
                # If we've reached the end of the settings block, stop reading fields
                else:
                    fields = False
                    # Parse the 'fields' setting to get the data type
                    dt = parse_fields(fields_text['fields'])
                    fields_text['data'] = np.zeros([1], dtype=dt)
                    break

        # Read the remaining data from the file using the parsed data type
        dt = parse_fields(fields_text['fields'])
        data = np.fromfile(f, dt)
        fields_text.update({'data': data})
        fields_text.update({'filename': os.path.basename(filename)})
        return fields_text


def organize_single_trodes_export(dir_path, skip_raw_group0=True):
    """
    Organizes Trodes data files in a given directory. The data is stored in a dictionary.
    The key is the penultimate (second to last) part of the file name (i.e., the part before the last dot in the file name).
    The values in the dictionary are the parsed data from the Trodes files.

    Args:
        dir_path (str): The path to the directory containing the Trodes files.
        skip_raw_group0(bool): To skip the "raw_group0" file which contains the raw signal which uses a lot of memory
    Returns:
        dict: A dictionary with organized Trodes file data.
    """
    # Initialize dictionary to store results
    result = {}

    # Iterate over all files in the directory
    for file_name in os.listdir(dir_path):

        if skip_raw_group0 and "raw_group0" in file_name:
            continue
        # Attempt to parse each file and store the data in the dictionary
        try:
            # Extract second to last part of the file name
            sub_dir_name = file_name.rsplit('.', 2)[-2]
            # Parse Trodes file and store the data
            result[sub_dir_name] = read_trodes_extracted_data_file(os.path.join(dir_path, file_name))

        # Skip files that cause errors during parsing
        except Exception as e:
            print(f"Skipping file {file_name} due to error: {e}")
            continue

    return result


def organize_all_trodes_export(dir_path):
    """
    Organize Trodes files in subdirectories based on prefix and suffix of the subdirectory.
    The function creates a dictionary with subdirectory prefix and suffix as keys,
    and the output of `organize_trodes_files_by_suffix` as values.

    Args:
        dir_path (str): Path of the directory to process.

    Returns:
        dict: Nested dictionary with keys as subdirectory prefix and suffix and values
        containing data obtained from the `organize_trodes_files_by_suffix` function.
    """
    result = defaultdict(dict)

    for sub_dir_name in os.listdir(dir_path):
        # Construct the full path to the subdirectory
        sub_dir_path = os.path.join(dir_path, sub_dir_name)
        # Process only if it's a directory
        if os.path.isdir(sub_dir_path):
            try:
                # Split the subdirectory name by dots to extract prefix and suffix
                sub_dir_name_parts = sub_dir_name.split('.')
                sub_dir_name_prefix = sub_dir_name_parts[0]
                sub_dir_name_suffix = sub_dir_name_parts[-1]
                # Organize the Trodes files in the subdirectory and store the results
                result[sub_dir_name_prefix][sub_dir_name_suffix] = organize_single_trodes_export(sub_dir_path)
            except Exception as e:
                print(f"Error processing subdirectory {sub_dir_path}: {e}")
                continue

    return result


##########################
##########################
##########################


def get_key_with_substring(input_dict, substring="", return_first=True):
    """
    Returns keys from a dictionary that contains a specified substring.

    Args:
        input_dict (dict): The dictionary to search.
        substring (str, optional): The substring to search for in the keys. Defaults to "".
        return_first (bool, optional): If True, returns the first key that contains the substring.
            If False, returns a list of all keys that contain the substring. Defaults to True.

    Returns:
        str or list: A key or a list of keys from the input dictionary that contains the substring.
            If the substring is an empty string, returns the first key from the dictionary
            or a list of all keys, depending on the value of 'return_first'.
            If the substring is itself a key in the dictionary, returns the substring.
            If no keys contain the substring, returns an empty string or list.
    """
    # Find all keys that contain the substring
    keys_with_substring = [key for key in input_dict.keys() if substring in key]

    # If the substring is itself a key in the dictionary, return it
    if substring in keys_with_substring:
        return substring

    # If 'return_first' is True, return the first key that contains the substring
    # If no keys contain the substring, return an empty string
    elif return_first and keys_with_substring:
        return keys_with_substring[0]

    # If 'return_first' is False, return a list of all keys that contain the substring
    # If no keys contain the substring, return an empty list
    else:
        return keys_with_substring


def get_all_file_suffixes(file_name):
    """
    Creates a string of the suffixes of a file name that's joined together by ".".
    Suffixes will be all the parts of the file name that follow the first ".".
    Example: If file_name is "file.txt.zip.asc", the output will be "txt.zip.asc".

    Args:
        file_name (str): Name of the file.

    Returns:
        str: A string of all the suffixes joined by ".", or a single "." if no suffixes exist.
    """
    # Extract all the suffixes in the file name using pathlib.Path().suffixes
    # This will return a list of suffixes.
    suffixes = pathlib.Path(file_name).suffixes

    # Strip any periods from the beginning or end of each suffix
    stripped_suffixes = [suffix.strip(".") for suffix in suffixes]

    # If there are suffixes, join them together with periods and return the result
    if stripped_suffixes:
        return ".".join(stripped_suffixes)

    # If there are no suffixes (i.e., the file name is just "."), return a single period
    else:
        return "."


def update_trodes_file_to_data(file_path, file_to_data=None):
    """
    Extracts the data and metadata from a Trodes recording file and stores it in a dictionary.
    The dictionary keys are file names, and the values are sub-dictionaries of data and metadata points.

    Args:
        file_path (str): Path to the Trodes recording file. The path can be relative or absolute.
        file_to_data (dict, optional): An existing dictionary to which the data will be added. If None, a new dictionary is created. Defaults to None.

    Returns:
        dict: A dictionary where each key is a file name and each value is a sub-dictionary containing data and metadata from the Trodes recording file.

    Raises:
        A warning if the Trodes recording file cannot be processed.
    """
    # Create a new dictionary if none is provided
    if file_to_data is None:
        file_to_data = defaultdict(dict)

    # Get just the file name to use as the key
    file_name = os.path.basename(file_path)

    # Get the absolute file path as metadata
    absolute_file_path = os.path.abspath(file_path)

    try:
        # Read the Trodes recording file data
        trodes_recording = parse_exported_file(absolute_file_path)

        file_prefix = get_all_file_suffixes(file_name)
        print(f"file prefix: {file_prefix}")

        # Store the data and metadata in the dictionary
        file_to_data[file_prefix] = trodes_recording
        file_to_data[file_prefix]["absolute_file_path"] = absolute_file_path
        return file_to_data
    except:
        # Issue a warning if the file cannot be processed
        warnings.warn(f"Cannot process {absolute_file_path}")
        return None


def get_all_trodes_data_from_directory(parent_directory_path="."):
    """
    Extracts data and metadata from all Trodes files in a given directory and its subdirectories.
    The data is organized in a dictionary where the keys are directory names and the values are
    dictionaries with file names as keys and associated data/metadata as values.

    Args:
        parent_directory_path (str): Path to the parent directory containing the Trodes recording files.
                                     This can be a relative or absolute path. Defaults to the current directory.

    Returns:
        dict: A dictionary where each key is a directory name and each value is a sub-dictionary containing
              file names as keys and corresponding data/metadata from Trodes recording files as values.
    """
    directory_to_file_to_data = defaultdict(dict)

    # Iterate over all items in the parent directory
    for item in os.listdir(parent_directory_path):
        item_path = os.path.join(parent_directory_path, item)

        # If the item is a directory, use its name as a key
        if os.path.isdir(item_path):
            current_directory_name = os.path.basename(item_path)
        # If the item is a file, use the parent directory name as a key
        else:
            current_directory_name = "."

        directory_prefix = get_all_file_suffixes(current_directory_name)
        current_directory_path = os.path.join(parent_directory_path, current_directory_name)

        # Iterate over all files in the current directory
        for file_name in os.listdir(current_directory_path):
            file_path = os.path.join(current_directory_path, file_name)
            if os.path.isfile(file_path):
                # Update the dictionary with data/metadata from the current file
                current_directory_to_file_to_data = update_trodes_file_to_data(file_path=file_path,
                                                                               file_to_data=directory_to_file_to_data[
                                                                                   current_directory_name])
                # If the file was processed successfully, update the main dictionary
                if current_directory_to_file_to_data is not None:
                    directory_to_file_to_data[directory_prefix] = current_directory_to_file_to_data

    return directory_to_file_to_data


def main():
    """
    Main function that runs when the script is run
    """


if __name__ == '__main__':
    main()