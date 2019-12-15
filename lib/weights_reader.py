import numpy as np


def read_weights(filename: str):
    data = np.genfromtxt(filename, delimiter=',')
    return data
