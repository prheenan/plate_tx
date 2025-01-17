"""
General utilities
"""
import string
import pandas
import numpy as np

def matrix_to_plate_df(matrix,n_rows=None,n_cols=None):
    """

    :param matrix: matrix to convert
    :param n_rows:  number of rows; force to be at least this size
    :param n_cols: number of columns; force to be at least this size
    :return: matrix represented as a dataframe, with index being row names
    and column being column names
    """
    size_rows, size_cols = matrix.shape
    if n_rows is None:
        n_rows = size_rows
    if n_cols is None:
        n_cols = size_cols
    if size_cols < n_cols or size_rows < n_rows:
        matrix = np.pad(array=matrix,
                        pad_width=((0, max(0, n_rows - size_rows)),
                                   (0, max(0, n_cols - size_cols))),
                        mode="constant",
                        constant_values=np.nan)
    size_rows_final, size_cols_final = matrix.shape
    rows = labels_rows(n=size_rows_final)
    cols = labels_cols(n=size_cols_final)
    return pandas.DataFrame(matrix,columns=cols,index=rows)

def plate_to_flat_df(df_plate):
    """

    :param df_plate:  output of matrix_to_plate_df
    :return: flat dataframe version of df_plae
    """
    df_flat = pandas.melt(df_plate.reset_index(), id_vars="Row",
                          var_name="Column")
    df_flat.insert(loc=0, column="Well",
                   value=[f"{r}{c}" for r, c in zip(df_flat["Row"],
                                                    df_flat["Column"])])
    df_flat.sort_values(by="Well", inplace=True)
    return df_flat

def flat_to_plate_df(df_flat,col_value="Value"):
    """

    :param df_flat: e.g. output of plate_to_flat_df
    :param col_value: value column
    :return: e.g. output of matrix_to_plate_df
    """
    return pandas.pivot_table(df_flat[["Row","Column",col_value]],
                              index="Row",columns="Column",values=col_value)

def labels_rows(n,preamble_first_26=None,preamble_next_26=None):
    """

    :param n: number  of rows
    :param preamble_first_26: second letter goes A-Z; starts with this (e.g., AA)
    :param preamble_next_26:  second letter goes A-Z; starts with this (e.g., BA)
    :return: list of n row labels
    """
    if preamble_first_26 is None:
        if n <= 26:
            # 384W and less typically don't have preambles
            preamble_first_26 = ""
        else:
            preamble_first_26 = "A"
    if preamble_next_26 is None:
        preamble_next_26 = "B"
    to_ret = [ preamble_first_26 + s for s in string.ascii_uppercase[:n]]
    if n >= 26:
        to_ret += [preamble_next_26 + a for a in string.ascii_uppercase[:n-26]]
    return to_ret

def labels_cols(n,leading_zero=None):
    """

    :param n: number of columns
    :param leading_zero: if true, add a leading zero (e.g., 01, 02), otherwise
    don't
    :return: list of n labels
    """
    if leading_zero is None:
        if n <= 9:
            leading_zero = False
        else:
            leading_zero = True
    return [f"{i:02d}" if leading_zero else f"{i:d}"
            for i in range(1,1+n)]

def plate_to_well_dict():
    return {
        "6": {'n_rows': 2, 'n_cols': 3},  # 6 well
        "12": {'n_rows': 3, 'n_cols': 4},  # 12 well
        "24": {'n_rows': 4, 'n_cols': 6},  # 24 well
        "48": {'n_rows': 6, 'n_cols': 8},  # 48 well
        "96": {'n_rows': 8, 'n_cols': 12},  # 96 well
        "384": {'n_rows': 16, 'n_cols': 24},  # 384 well
        "1536": {'n_rows': 32, 'n_cols': 48},  # 1536 well
        "3456": {'n_rows': 48, 'n_cols': 72},  # 3456 well
    }

def plate_row_cols(p):
    """

    :param p: plate name (e.g., 384)
    :return:  2-tuple, <number of rows, number of columns>
    """
    dict_v = plate_to_well_dict()
    if p in dict_v:
        pass
    elif str(p) in dict_v:
        p = str(p)
    else:
        raise ValueError(f"Didn't understand plate type {p}")
    return dict_v[p]["n_rows"], dict_v[p]["n_cols"]

def plate_rows(p):
    """

    :param p: see plate_row_cols
    :return: number of rows for plate
    """
    return plate_row_cols(p)[0]

def plate_cols(p):
    """

    :param p: see plate_row_cols
    :return: number of columns for plate
    """
    return plate_row_cols(p)[1]
