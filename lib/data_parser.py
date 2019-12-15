import numpy as np


def parse_target(row_data: str, target_size: int):
    # clean up input str
    row_data_cleaned: str = clean_row_data(row_data)
    # Convert the str to list
    split_row = row_data_cleaned.split(',')
    target = split_row[0]
    targets = np.zeros(target_size) + 0.01
    targets[int(target)] = 0.999
    return targets


def parse_target_number(row_data: str):
    # clean up input str
    row_data_cleaned: str = clean_row_data(row_data)
    # Convert the str to list
    split_row = row_data_cleaned.split(',')
    target = split_row[0]
    return int(target)


def parse_scale_image_data(row_data: str):
    # clean up input str
    row_data_cleaned: str = clean_row_data(row_data)
    # Convert the str to list
    split_row = row_data_cleaned.split(',')
    # Scale the numbers by dividing 255 and times 0.99
    # We need the numbers between 0.01 and 0.999, no zeros and ones
    image_data_array = (np.asfarray(split_row[1:], float) / 255.0 * 0.999) + 0.01
    return image_data_array

def parse_scale_image_data_without_target(row_data: str):
    # clean up input str
    row_data_cleaned: str = clean_row_data(row_data)
    # Convert the str to list
    split_row = row_data_cleaned.split(',')
    # Scale the numbers by dividing 255 and times 0.99
    # We need the numbers between 0.01 and 0.999, no zeros and ones
    image_data_array = (np.asfarray(split_row[0:], float) / 255.0 * 0.999) + 0.01
    return image_data_array

def clean_row_data(row_data: str) -> str:
    # Remove the double quotes at front of the str
    row_remove_front_quote: str = row_data.strip('"')
    # Remove the double quote and \n at the end of the str
    row_remove_end_quote: str = row_remove_front_quote.strip('"\n')
    return row_remove_end_quote


def parse_image_data(row_data: str):
    # clean up input str
    row_data_cleaned: str = clean_row_data(row_data)
    # Convert the str to list
    split_row = row_data_cleaned.split(',')
    # Scale the numbers by dividing 255 and times 0.99
    # We need the numbers between 0.01 and 0.999, no zeros and ones
    image_data = np.asfarray(split_row[1:]).reshape((28,28))
    return image_data
