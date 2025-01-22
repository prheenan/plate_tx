"""
Defining how to read plate files
"""
import pandas
import utilities


def return_plate_no_header(_,matrix):
    """

    :param _: header (ignored)
    :param matrix:  2D matrix
    :return: matrix-as-plate
    """
    return utilities.matrix_to_plate_df(matrix)


PLATE_PARAMS = {
    "default": {'kw_read': {},
                'f_header_matrix_to_plate': return_plate_no_header},

}


def _read_xlsx_as_dict_of_df(file_name,**kw):
    """

    :param file_name: XLSX file
    :param kw: passed to pandas.read_excel
    :return: dictionary of data frames to parse for data
    """
    return pandas.read_excel(file_name,sheet_name=None,header=None,**kw)

def _read_text_as_dict_of_df(file_name,**kw):
    """

    :param file_name: CSV file
    :param kw: passed to pandas.read_csv
    :return: dictionary giving Value ad key and dataframe as value
    """
    # just one here
    return {"Sheet1":pandas.read_csv(file_name,header=None,**kw)}


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

def _parse_plate_here(df,plate_start_row,plate_start_col):
    # need to find where the plate ends, which is denote by either:
    # (1) end of dataframe
    # (2) start of next plate
    # (3) missing row/column labels (??)
    max_rows_cols = max(utilities.plate_to_well_dict().items(),
                        key=lambda e: int(e[0]))[1]
    n_max_rows, n_max_cols = max_rows_cols["n_rows"], max_rows_cols["n_cols"]
    row_options = [utilities.labels_rows(n_max_rows),
                   utilities.labels_rows(n_max_rows,
                                         preamble_first_26="")]
    col_options = [utilities.labels_cols(n_max_cols,leading_zero=False),
                   utilities.labels_cols(n_max_cols, leading_zero=True)]
    ordered_rows = [ {e[i] for e in row_options} for i in range(n_max_rows)]
    ordered_cols = [ {int(e[j]) for e in col_options} for j in range(n_max_cols)]
    # advance once row and one column (we know we have at least one well of data)
    plate_end_col = plate_start_col + 1
    plate_end_row = plate_start_row + 1
    n_df = len(df)
    n_col = len(df.columns)
    i_col = 1
    while plate_end_col < n_col and i_col < n_max_cols and \
            (int_or_none(df.iloc[plate_start_row - 1, plate_end_col]) in ordered_cols[i_col]):
        plate_end_col += 1
        i_col += 1
    j_row = 1
    while plate_end_row < n_df and j_row < n_max_rows and \
            (df.iloc[plate_end_row, plate_start_col - 1] in ordered_rows[j_row]):
        plate_end_row += 1
        j_row += 1
    return plate_end_row,plate_end_col

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
                # then parse the plate here
                plate_start_row = i
                plate_start_col = j + 1
                plate_end_row, plate_end_col = \
                    _parse_plate_here(df, plate_start_row, plate_start_col)
                plate_header = df.iloc[previous_end_row:plate_start_row]
                plate_data = df.iloc[plate_start_row:plate_end_row,
                             plate_start_col:plate_end_col]
                plates.append([plate_header,plate_data])
                j = plate_end_col
                previous_end_row = plate_end_row
            else:
                j += 1
        # POST: j = n_col-1, so at the end
        if plate_start_row is None:
            # didn't find a plate on any column, move along
            i += 1
        else:
            # did find all the plates, so can move to the end of the plates
            i = max(i+1,previous_end_row)
    return plates


def plate_to_flat(file_name,file_type="default"):
    """

    :param file_name: file name to read in
    :param file_type: type of file this is to parse
    :return:  depends on file type
    """
    if file_type not in PLATE_PARAMS:
        raise ValueError(f"Did not understand {file_type}")
    params = PLATE_PARAMS[file_type]
    kw_read = params["kw_read"]
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        dict_of_df = _read_xlsx_as_dict_of_df(file_name,**kw_read)
    else:
        dict_of_df = _read_text_as_dict_of_df(file_name,**kw_read)
    parsed_headers_plates = { k:_parse_all_plates(v)
                              for k,v in dict_of_df.items()}
    # for now, jut convert the plates into dataframes and ignore the header
    f_to_plate = params['f_header_matrix_to_plate']
    flat_files = {k:[f_to_plate(header,matrix) for header,matrix in header_plates]
                  for k,header_plates in parsed_headers_plates.items()}
    return flat_files
