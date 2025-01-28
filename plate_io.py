"""
Defining how to read plate files
"""
import re
from io import StringIO
import pandas
import numpy as np
import utilities

def return_plate_no_header(_,matrix):
    """

    :param _: header (ignored)
    :param matrix:  2D matrix
    :return: matrix-as-plate
    """
    return utilities.matrix_to_plate_df(matrix)

def return_plate_or_none_if_all(_,matrix,f_test):
    """

    :param _:
    :param matrix:
    :param f_test:
    :return:
    """
    f_all_vect = np.vectorize(f_test)
    values = matrix.to_numpy()
    matrix_vect = f_all_vect(values)
    if np.all(matrix_vect):
        return None
    return utilities.matrix_to_plate_df(matrix)


def save_all_plates(plate_df_colors,file_name,index=True):
    """

    :param plate_df_colors: either list of plates (CSV or XLSX) or dictionary going
     from sheet name to list of plates (XLSX only)
    :param file_name:  file to save
    :param index: if true
    :return:  nothing
    """
    if file_name.endswith(".csv"):
        if isinstance(plate_df_colors,dict):
            dfs = [ e for k in plate_df_colors
                    for e in plate_df_colors[k]]
        elif isinstance(plate_df_colors,list):
            # is a list
            dfs = plate_df_colors
        else:
            raise ValueError(f"Didn't understand how to save {type(plate_df_colors)}")
        dfs[0].to_csv(file_name, index=index)
        if len(dfs) > 1:
            for d in dfs[1:]:
                d.to_csv(file_name, index=index, mode="a")
    else:
        if isinstance(plate_df_colors,list):
            sheet_to_dfs = {"Sheet1":plate_df_colors}
        else:
            sheet_to_dfs = plate_df_colors
        # pylint: disable=abstract-class-instantiated
        with pandas.ExcelWriter(file_name, engine="openpyxl") as xlsx:
            for sheet,plates_sheet in sheet_to_dfs.items():
                i_row = 0
                for p in plates_sheet:
                    p.to_excel(xlsx, startrow=i_row,sheet_name=sheet,
                               index=index)
                    # add 1 for header
                    i_row += len(p) + 1


def return_plate_and_header(header,matrix):
    """

    :param header: header
    :param matrix: matrix
    :return: tuple of header and matrix-as-plate (See return_plate_no_header)
    """
    return header, return_plate_no_header(header,matrix)

def rows_until_plate_start_spaces(lines,offset=0,**kw):
    """
    conveinece function to find start of space-delimited plate

    :param lines: space-delimited lines
    :param offset:  offset from row
    :param kw: see rows_until_plate_start
    :return: see rows_until_plate_start
    """
    return rows_until_plate_start(lines,sep=r"\s",offset=offset,**kw)

def rows_until_plate_start(lines,sep=r",",offset=0,**kw):
    """

    :param lines: lines
    :param sep:  separarator
    :param offset: offset to return from the line index that matches
    :param kw: passed to rows_until_regex
    :return:  see rows_until_regex
    """
    regex = rf"""^
            (
             {sep}  # may have leading separatators
             [0-9]  # first label may or may not have the separator
            )?
            (
            {sep}+  # after that, must have separator
            [0-9]+
            )+      # at least two additional labels (but bounded above)
            {sep}*  # may have trailing separators
            $
            """
    return rows_until_regex(lines=lines,regex=regex, offset=offset,**kw)

def rows_until_plate_end_spaces(plate_start_rows,lines):
    """

    :param plate_start_rows: see  rows_until_plate_end
    :param lines: see rows_until_plate_end
    :return: see rows_until_plate_end
    """
    return  rows_until_plate_end(plate_start_rows,lines,sep=r"\s")

def rows_until_flat_end(plate_start_rows,lines,sep=r";"):
    """

    :param plate_start_rows: see  output of rows_until_plate_start
    :param lines:  all lines (headers)
    :param sep: separation string
    :return: rows of plate ends, list N
    """
    regex_still_plate = rf"""
                         ^
                         {sep}*      # possible separation
                         # well label is :
                         # (1 or two characters, like A or AF)
                         # (1 or two digits, like 1 or 01)
                         # e.g., Well AA01
                         [A-Z][A-Z]?[0-9][0-9]?
                         {sep}+      # separation
                         """
    return _rows_until_end_by_regex(lines, plate_start_rows, regex_still_plate)

def rows_until_plate_end(plate_start_rows,lines,sep=r","):
    """

    :param plate_start_rows: list, length N, of plate starts
    :param lines:  lines to consider
    :param sep:  separation
    :return: list, length N, of plate ends
    """
    regex_still_plate = rf"""
                         ^
                         {sep}*      # possible separation
                         [A-Z][A-Z]? # row label (1 or two characters, like A or AF)
                         {sep}+      # separation
                         """
    return _rows_until_end_by_regex(lines, plate_start_rows, regex_still_plate)

def _rows_until_end_by_regex(lines,plate_start_rows,regex_still_plate):
    """

    :param lines: lines to search
    :param plate_start_rows: rows where plate starts, length N
    :param regex_still_plate: still a plate while this is true (or not at end)
    :return:  length N list of row ends
    """
    pattern = re.compile(regex_still_plate,re.VERBOSE)
    n_lines = len(lines)
    end_rows = []
    for start_line in plate_start_rows:
        end_row = start_line + 1
        while (end_row < n_lines) and pattern.match(lines[end_row]):
            end_row += 1
        # POST: at the end of the plate or end of the file
        end_rows.append(end_row)
    return end_rows

#pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
def flat_converter(plate,start_row,col_rename,still_row_if,use_col_if=None,
                   start_col=None):
    """

    :param plate: dataframe
    :param start_row:  what the start cell should look like
    :param col_rename: function, takes column index and name; returns new column
    name
    :param still_row_if: function, takes column index and name; use the
    row if matches
    :param use_col_if:  function, takes column index and name; use the
    column if this is true
    :param start_col: start column
    :return:
    """
    if start_col is None:
        start_col = 0
    start_header = 0
    pattern_still = re.compile(still_row_if)
    plates_to_return = []
    if isinstance(start_col, str):
        # find the start column
        start_col_index = None
        for i, row in plate.iterrows():
            for j, e in enumerate(row):
                if e == start_col:
                    start_col_index = j
        if start_col_index is None:
            raise ValueError(f"Couldn't find column like {start_col}")
    else:
        # not a string, ass
        start_col_index = start_col
    i = 0
    while i < len(plate):
        if plate.iloc[i,start_col_index] == start_row:
            start_idx = i
            while(i < len(plate) and pattern_still.match(plate.iloc[i,start_col_index])):
                i += 1
            # POST: at end of file or no longer part of plate
            header = plate.iloc[start_header:start_idx,:]
            df_flat = plate.iloc[start_idx:i,:]
            df_flat.columns = [col_rename(i_c,c)
                               for i_c,c in enumerate(plate.iloc[start_idx-1,:])]
            if use_col_if is not None:
                cols_to_use = [c for i_c,c in enumerate(df_flat.columns)
                               if use_col_if(i_c,c) ]
                df_flat = df_flat[["Well"] + cols_to_use]
            else:
                cols_to_use = df_flat.columns
            if "Well" not in df_flat.columns:
                raise ValueError("Must specify well")
            df_flat["Well"] = df_flat["Well"].transform(utilities.sanitize_well)
            df_flat["Row"] = df_flat["Well"].transform(utilities.row_from_well)
            df_flat["Column"] = df_flat["Well"].transform(utilities.column_from_well)
            for c in cols_to_use:
                df_plate = utilities.flat_to_plate_df(df_flat[["Row","Column",c]],col_value=c)
                # reset so that the rows and indcies are just in the data
                df_plate_reset = df_plate.reset_index().T.reset_index().T
                # adjust the header columns
                max_len = min(len(header.columns),len(df_plate_reset.columns))
                header.columns = df_plate_reset.columns[:max_len]
                plates_to_return.append(pandas.concat([header,df_plate_reset]))
            start_header = i
        else:
            i += 1
    return pandas.concat(plates_to_return)



def rows_until_regex(lines,regex,offset=0):
    """

    :param lines: list of N lines
    :param regex:  regex to search
    :param offset: offset return value relative to this
    :return: how many lines until occurences of regex (e.g., if on first
    line then returns 1, second line 2, etc). List.
    """
    pattern = re.compile(regex,re.VERBOSE)
    to_ret = []
    for i,l in enumerate(lines):
        if pattern.match(l):
            # index i is this regex, so e.g. i+1 would be number of rows to it
            to_ret.append(i+offset)
    return to_ret



def _read_xlsx_as_dict_of_df(file_name,header=None,**kw):
    """

    :param file_name: XLSX file
    :param kw: passed to pandas.read_excel
    :return: dictionary of data frames to parse for data
    """
    return pandas.read_excel(file_name,sheet_name=None,header=header,**kw)

def join_values(values):
    """

    :param values: list of strings to join
    :return: values joined as string
    """
    return "".join(values)

#pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def _read_text_as_dict_of_df(file_name,f_header_until=None,f_join_plate=None,
                             f_plate_until=None,header=None,
                             encoding="utf-8",engine="python",**kw):
    """

    :param file_name: CSV file
    :param kw: passed to pandas.read_csv
    :return: dictionary giving Value ad key and dataframe as value
    """
    # just one here
    with open(file_name, 'r', encoding=encoding) as f:
        lines = f.readlines()
    if f_join_plate is None:
        f_join_plate = join_values
    if f_header_until is None:
        plate_start_rows = []
        plate_end_rows = []
    else:
        plate_start_rows = f_header_until(lines=lines)
        plate_end_rows = f_plate_until(plate_start_rows=plate_start_rows,lines=lines)
    if len(plate_start_rows) > 0:
        df_header_arr = []
        lines_file_arr = []
        previous_n = 0
        for n_row_i,n_row_f in zip(plate_start_rows,plate_end_rows):
            df_header_arr.append(pandas.DataFrame(lines[previous_n:n_row_i]))
            lines_file_arr.append(lines[n_row_i:n_row_f])
            previous_n = n_row_f
    else:
        df_header_arr = [pandas.DataFrame()]
        lines_file_arr = [lines]
    df_to_ret = []
    for lines_file,df_header_i in zip(lines_file_arr,df_header_arr):
        try:
            df_no_header = pandas.read_csv(StringIO(f_join_plate(lines_file)),
                                           header=header,on_bad_lines="warn",
                                           engine=engine,**kw)
        except pandas.errors.EmptyDataError:
            df_no_header = pandas.DataFrame()
        df_to_ret.append(pandas.concat([df_header_i,df_no_header],ignore_index=True))
    return {"Sheet1":pandas.concat(df_to_ret,ignore_index=True)}


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
                # header starts one row back
                plate_header = df.iloc[previous_end_row:plate_start_row-1]
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

def read_file(file_name,kw_read_xlsx=None,kw_read_csv=None,**kw):
    """

    :param file_name: name of file to read
    :param kw_read_xlsx:  xlsx-specific keywords (if xlsx file)
    :param kw_read_csv: csv-specicic keywords (if csv file)
    :param kw:  keywords for any read
    :return: file as dataframe
    """
    if kw_read_xlsx is None:
        kw_read_xlsx = {**kw}
    if kw_read_csv is None:
        kw_read_csv = {**kw}
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        dict_of_df = _read_xlsx_as_dict_of_df(file_name,**kw_read_xlsx)
    else:
        dict_of_df = _read_text_as_dict_of_df(file_name,**kw_read_csv)
    return dict_of_df

def read_plate_dict(file_name, file_type="DEFAULT PLATE"):
    """

    :param file_name: file name to read in
    :param file_type: type of file this is to parse
    :return:  depends on file type
    """
    if file_type.upper() not in (PLATE_PARAMS | FLAT_PARAMS):
        raise ValueError(f"Did not understand {file_type}")
    params = (PLATE_PARAMS | FLAT_PARAMS)[file_type.upper()]
    kw_read_xlsx = params["kw_read_xlsx"]
    kw_read_csv = params["kw_read_csv"]
    dict_of_df = read_file(file_name,
                           kw_read_xlsx=kw_read_xlsx,kw_read_csv=kw_read_csv)
    convert_f = params["convert_plate_function"] \
                if "convert_plate_function" in params else lambda x: x
    parsed_headers_plates = { k:_parse_all_plates(convert_f(v))
                              for k,v in dict_of_df.items()}
    # for now, jut convert the plates into dataframes and ignore the header
    f_to_plate = params['f_header_matrix_to_plate']
    flat_files = {k:[f_to_plate(header,matrix) for header,matrix in header_plates]
                  for k,header_plates in parsed_headers_plates.items()}
    # don't return plates which are None
    flat_files = { k:[e for e in list_v if e is not None]
                   for k,list_v in flat_files.items()}
    return flat_files

def matrix_to_video_dict(video):
    """

    :param video: like times X height X width X C, where C is 1 for greyscale or 3 for RGB
    :return: dictionary, keys are frame index and values are matrixes like heightXwidthXC
    """
    times = video.shape[0]
    is_rgb = len(video.shape) == 4 and video.shape[-1] == 3
    if is_rgb:
        sheets = {f"Time {i:04d}":
                      [utilities.matrix_to_plate_df(video[i, :, :, j])
                       for j in range(3)]
                  for i in range(times)}
    else:
        sheets = {f"Time {i:04d}": \
                      [utilities.matrix_to_plate_df(video[i, :, :,0])]
                  for i in range(times)}
    return sheets

def file_to_video(file_name,file_type="DEFAULT PLATE",is_rgb=False):
    """

    :param file_name: either XLSX with R,G,B plates per sheet (time)
    or CSV with RGB for time 0, then RGB for  time 1, etc
    :param file_type: what type of file it is
    :param is_rgb: if true, is RGB
    :return: array like < times, height, width, rgb >
    """
    plate_dict = read_plate_dict(file_name=file_name,file_type=file_type)
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        # stack the sheets directly
        if is_rgb:
            # each sheet has r,g,b plates
            frames_group = plate_dict.values()
        else:
            # flatten them all
            frames_group = [e for v in plate_dict.values()
                            for e in v]
    else:
        frames = plate_dict[list(plate_dict.keys())[0]]
        if is_rgb:
            # split into rgb
            frames_group = [ [frames[i * 3], frames[i * 3 + 1], frames[i * 3 + 2]]
                             for i in range(len(frames)//3)]
        else:
            # use all the frames directly
            frames_group = frames
    if is_rgb:
        time_rgb = np.array([np.dstack(f) for f in frames_group])
    else:
        time_rgb = np.array(frames_group)
    return time_rgb

# name plated types are from
# https://sciencecloud-preview.my.site.com/s/article/Assay-Reader-Plate-Formats
_KW_CSV = {"sep": r",",
           "f_header_until": rows_until_plate_start,
           "f_plate_until": rows_until_plate_end}

FLAT_PARAMS = {
    "DEFAULT FLAT": {
        'kw_read_xlsx': {},
        'kw_read_csv': {"sep": r",",
                        # plate starts at A01
                        "f_header_until": lambda *args, **kw:
                        rows_until_regex(*args, regex=r",\s*Well\s*,",
                                         offset=1, **kw),
                        "f_plate_until": lambda *args, **kw:
                        _rows_until_end_by_regex(*args, **kw,
                                                 regex_still_plate=r",\s*[A-Z][A-Z]?\d+\s*,"),
                        },
        'convert_plate_function': lambda plate: \
            flat_converter(plate, start_row="A01", start_col="Well",
                           col_rename=lambda i,c: c,
                           still_row_if="[A-Z][A-Z]?[0-9][0-9]?",
                           # Value 0 would be the plate
                           use_col_if=lambda i, c: c not in ["Well","Row","Column"]),
        'f_header_matrix_to_plate': return_plate_no_header
    },
    "BMG PHERASTAR": {
            'kw_read_xlsx': {},
            'kw_read_csv': {"sep": ";",
                            # plate starts at A01
                            "f_header_until": lambda *args,**kw:
                                rows_until_regex(*args,regex=r"^\s*A0?1;\s+",**kw),
                            "f_plate_until": rows_until_flat_end,
                            },
            'convert_plate_function': lambda plate: \
                flat_converter(plate,start_row="A01",
                               col_rename=lambda i,_: f"Value {i}" if i != 0 else "Well",
                               still_row_if="[A-Z][A-Z]?[0-9][0-9]?",
                               use_col_if=lambda i,c: c != "Well"),
            'f_header_matrix_to_plate': return_plate_no_header
    },
    "DELFIA ENVISION FLAT": {
        'kw_read_xlsx': {},
        'kw_read_csv': {"sep": r"\s",
                        # plate starts at A01
                        "f_header_until": lambda *args, **kw:
                        rows_until_regex(*args, regex=r"^Plate\s+Well\s+",offset=0, **kw),
                        "f_plate_until": lambda *args,**kw:
                        _rows_until_end_by_regex(*args,**kw,
                                                 regex_still_plate=r"^\d+\s+[A-Z][A-Z]?\d+\s+"),
                        },
        'convert_plate_function': lambda plate: \
            flat_converter(plate, start_row="A01",start_col="Well",
                           col_rename=lambda i, c: f"Value {i}" if c != "Well" else "Well",
                           still_row_if="[A-Z][A-Z]?[0-9][0-9]?",
                           # Value 0 would be the plate
                           use_col_if=lambda i, c: c not in ["Well","Value 0"]),
        'f_header_matrix_to_plate': return_plate_no_header
    }
}

PLATE_PARAMS = {
    "PLATE_AND_HEADER": {'kw_read_xlsx': {},
                          'kw_read_csv': {},
                          'f_header_matrix_to_plate': return_plate_and_header},
    "DEFAULT PLATE": {'kw_read_xlsx': {},
                      'kw_read_csv': {},
                      'f_header_matrix_to_plate': return_plate_no_header},
    "HCS CELLOMICS" : {'kw_read_xlsx': {},
                       'kw_read_csv': {"sep": r";",
                                       "f_header_until": lambda *args,**kw: \
                                            rows_until_plate_start(*args,sep=r";",**kw),
                                       "f_plate_until": lambda *args,**kw: \
                                           rows_until_plate_end(*args,sep=r";",**kw),
                                       },
                        'f_header_matrix_to_plate': return_plate_no_header
            },
    # ENVISION has a special matrix which is all "-" which we want to ignore
    "DELFIA ENVISION" : {
            'kw_read_xlsx': {},
            'kw_read_csv': _KW_CSV,
            'f_header_matrix_to_plate': lambda *args, **kw:\
                return_plate_or_none_if_all(*args,f_test = lambda x: str(x).strip() == "-",**kw)
    },
    "ANALYST GT": { 'kw_read_xlsx': {},
                    'kw_read_csv': {"sep":"\t",
                                   "f_header_until": rows_until_plate_start_spaces,
                                   "f_plate_until": rows_until_plate_end_spaces,
                                    },
                   'f_header_matrix_to_plate': return_plate_no_header},
    "BMG LABTECH": {  'kw_read_xlsx': {},
                      'kw_read_csv': _KW_CSV,
                      'f_header_matrix_to_plate': return_plate_no_header
                      },
    "CLARIOSTAR": {  'kw_read_xlsx': {},
                      'kw_read_csv': {"sep":r"\s+",
                                      # add a single leading zero to the first line
                                      'f_join_plate':lambda x: \
                                          "".join(["0" + l if i ==0 else l
                                                   for i,l in enumerate(x)]),
                                   "f_header_until": rows_until_plate_start_spaces,
                                   "f_plate_until": rows_until_plate_end_spaces
                                      },
                      'f_header_matrix_to_plate': return_plate_no_header
                      },
}
