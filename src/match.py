import math
import numpy as np
import pandas as pd


def check_availability(x1, x2):
    # print float(len(set(x1).intersection(set(x2))))
    if float(len(set(x1).intersection(set(x2))))/len(x1) > 0:
        return True, float(len(set(x1).intersection(set(x2))))
    return False, float(len(set(x1).intersection(set(x2))))
