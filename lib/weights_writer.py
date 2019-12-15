import numpy as np
import os
from os import path
from datetime import datetime


def write_file(filename: str, content):
    backup_file(filename)
    np.savetxt(filename, content, fmt='%10.7f', delimiter=',')


def backup_file(filename: str):
    if path.exists(filename):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        (old_filename, ext) = os.path.splitext(filename)
        new_filename = old_filename + "-" + timestamp + ext
        os.rename(filename, new_filename)
