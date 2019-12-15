from typing import TextIO, List


def read_file(filename: str):
    data_file: TextIO = open(filename, 'r')
    data = data_file.readlines()
    data_file.close()
    length: int = len(data)
    print('LEN=', length)
    return data
