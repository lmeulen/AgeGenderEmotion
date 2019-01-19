"""
Module with utility functions for arrays
"""
def scale(array):
    """
    Scale values in x from [0..255] to [-1, 1]
    :param array: array to scale
    :return: rescaled array (float values)
    """
    array = array.astype('float32')
    array = array / 255.0
    array = array - 0.5
    array = array * 2.0
    return array