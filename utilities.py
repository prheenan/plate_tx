"""
General utilities
"""
import string

def labels_rows(n,preamble_first_26="A",preamble_next_26="B"):
    """

    :param n: number  of rows
    :param preamble_first_26: second letter goes A-Z; starts with this (e.g., AA)
    :param preamble_next_26:  second letter goes A-Z; starts with this (e.g., BA)
    :return: list of n row labels
    """
    to_ret = [ preamble_first_26 + s for s in string.ascii_uppercase[:n]]
    if n >= 26:
        to_ret += [preamble_next_26 + a for a in string.ascii_uppercase[:n-26]]
    return to_ret

def labels_cols(n,leading_zero=True):
    """

    :param n: number of columns
    :param leading_zero: if true, add a leading zero (e.g., 01, 02), otherwise
    don't
    :return: list of n labels
    """
    return [f"{i:02d}" if leading_zero else f"{i:d}"
            for i in range(1,1+n)]
