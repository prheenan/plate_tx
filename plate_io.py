"""
Defining how to read plate files
"""
import pandas
import utilities

def _read_xlsx_as_dict_of_df(file_name):
    """

    :param file_name: XLSX file
    :return: dictionary of data frames to parse for data
    """
    return pandas.read_excel(file_name,sheet_name=None,header=None)

def _read_text_as_dict_of_df(file_name,_):
    """

    :param file_name: CSV file
    :param _:  this will be the file type eventually
    :return: dictionary giving Value ad key and dataframe as value
    """
    # just one here
    return {"Sheet1":pandas.read_csv(file_name,header=None)}


def is_start_row(v):
    """

    :param v: value
    :return:  true if a valid start row
    """
    return str(v).strip() in ["A", "AA"]

def int_or_none(v):
    """

    :param v: value
    :return:  converts to integer if possible otherwise returns none
    """
    try:
        return int(v)
    except ValueError:
        return None

def is_1(v):
    """

    :param v: value
    :return:  true if this is one
    """
    return int_or_none(v) == 1

#pylint: disable=too-many-locals
def _parse_all_plates(df):
    """

    :param df:  single value of _read_text_as_dict_of_df or _read_csv_as_dict_of_df
    :return: list; each element a 2-tuple like (header of plate, plate)
    """
    plates = []
    if len(df) < 2:
        # there can't be a plate here
        return plates
    # otherwise, try to parse all the individual plates
    # we want to divide the plate into two areas per plate

    # (1) before the plate
    # (2) the plate
    n_df = len(df)
    i = 1
    previous_end_row = i
    while i < n_df:
        row = df.iloc[i]
        j = 0
        n_col = len(row)
        plate_start_row = None
        while j < n_col-1:
            char_here = df.iloc[i, j]
            char_upper_right = df.iloc[i-1, j+1]
            if is_start_row(char_here) and is_1(char_upper_right):
                plate_start_row = i
                plate_start_col = j + 1
                # need to find where the plate ends, which is denote by either:
                # (1) end of dataframe
                # (2) start of next plate
                # (3) missing row/column labels (??)
                max_rows_cols = max(utilities.plate_to_well_dict().items(),
                                    key=lambda e: int(e[0]))[1]
                allowable_row_labels = \
                    (set(utilities.labels_rows(max_rows_cols["n_rows"])) |
                     set(utilities.labels_rows(max_rows_cols["n_rows"],preamble_first_26="")))
                allowable_col_labels = \
                    set(utilities.labels_cols(max_rows_cols["n_cols"],leading_zero=False)) | \
                    set(utilities.labels_cols(max_rows_cols["n_cols"],leading_zero=True))
                allowable_col_labels_int = {int(s) for s in allowable_col_labels}
                # advance once row and one column (we know we have at least one well of data)
                j += 1
                i += 1
                while j < n_col and \
                        (int_or_none(df.iloc[plate_start_row-1, j]) in allowable_col_labels_int):
                    j += 1
                while i < n_df and \
                        (df.iloc[i, plate_start_col-1] in allowable_row_labels):
                    i += 1
                plate_end_col = j
                plate_end_row = i
                plates.append([df.iloc[previous_end_row:plate_start_row],
                               df.iloc[plate_start_row:plate_end_row,
                               plate_start_col:plate_end_col]])
                if j < n_col:
                    # possible there is another plate still; reset the row search
                    i = previous_end_row
                else:
                    previous_end_row = min(n_df-1,plate_end_row + 1)
            else:
                j += 1
        if plate_start_row is None:
            i += 1
    return plates


def plate_to_flat(file_name,file_type=None):
    """

    :param file_name: file name to read in
    :param file_type: type of file this is to parse
    :return:  depends on file type
    """
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        dict_of_df = _read_xlsx_as_dict_of_df(file_name)
    else:
        dict_of_df = _read_text_as_dict_of_df(file_name,file_type)
    parsed_headers_plates = { k:_parse_all_plates(v) for k,v in dict_of_df.items()}
    # for now, jut convert the plates into dataframes and ignore the header
    flat_files = {k:[utilities.matrix_to_plate_df(plate)
                     for header,plate in header_plates]
                  for k,header_plates in parsed_headers_plates.items()}
    return flat_files
