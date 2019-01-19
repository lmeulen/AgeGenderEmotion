"""
Module with utility functions for arrays
"""
def scale(arr):
    """
    Scale values in x from [0..255] to [-1, 1]
    :param arr: array to scale
    :return: rescaled array (float values)
    """
    arr = arr.astype('float32')
    arr = arr / 255.0
    arr = arr - 0.5
    arr = arr * 2.0
    return arr