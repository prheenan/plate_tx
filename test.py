"""
All unit tests
"""
import unittest
import os
import tempfile

import pandas
import numpy as np
from numpy.random import randint
from skvideo.utils import rgb2gray
import video
import utilities
import plate_io
from plate_io import save_all_plates
import plate_tx

class MyTestCase(unittest.TestCase):
    """
    Tester class
    """
    def __init__(self,*args,**kwargs):
        """
        Intialization
        """
        super().__init__(*args,**kwargs)
        self.i_sub_test = 0

    @classmethod
    def setUpClass(cls):
        """

        :return: Nothing, sets up the class
        """
        # read in the video
        frames_test_rate = 200
        cls.video_matrix = video.read_video("data/mario-trim.mov")
        # https://en.wikipedia.org/wiki/Microplate#Formats_and_Standardization_efforts
        cls.resized_videos = {}
        cls.resized_videos_grey = {}
        for plate,plate_dict in utilities.plate_to_well_dict().items():
            n_rows,n_cols = plate_dict["n_rows"],plate_dict["n_cols"]
            resized = \
                video.resize_video(cls.video_matrix[::frames_test_rate],
                                   px_width=n_cols, px_height=n_rows,
                                   disable_tqdm=True)
            cls.resized_videos[plate] = resized
            cls.resized_videos_grey[plate] = rgb2gray(resized)


    def _check_plate_read_expected(self,file_name,sheet_to_matrices):
        """

        :param file_name: file name saved
        :param sheet_to_matrices: expectation: key is sheet name, value
        is list of matrices expected in order
        """
        sheet_to_dfs = plate_io.read_plate_dict(file_name)
        for sheet_name, matrices in sheet_to_matrices.items():
            with self.subTest(i=self.i_sub_test, msg=file_name):
                # should only have one file output
                assert len(sheet_to_dfs[sheet_name]) == len(matrices)
            self.i_sub_test += 1
            for i,m in enumerate(matrices):
                with self.subTest(i=self.i_sub_test, msg=file_name):
                    # make sure the data we read back in is correct
                    m_found = sheet_to_dfs[sheet_name][i].to_numpy()
                    assert (m_found == m).all()
                self.i_sub_test += 1

    #pylint: disable=too-many-locals, too-many-statements
    def test_98_pathological_file_io(self):
        """
         make sure a bunch of terrible examples give sensible outputs
        """
        self.i_sub_test = 0
        for save,suffix in [[lambda x,*args,**kw : x.to_excel(*args,**kw),".xlsx"],
                            [lambda x,*args,**kw: x.to_csv(*args,**kw),".csv"]]:
            with tempfile.NamedTemporaryFile(suffix=suffix) as f:
                # empty data frame should work
                save(pandas.DataFrame({}),f.name)
                with self.subTest(i=self.i_sub_test):
                    df_flat = plate_io.read_plate_dict(f.name)
                    assert len(df_flat["Sheet1"]) == 0
                self.i_sub_test += 1
                # try every size empty plate (just the column labels)
                for plate_size, plate_dict in utilities.plate_to_well_dict().items():
                    n_cols, n_rows = plate_dict["n_cols"], plate_dict["n_rows"]
                    cols = utilities.labels_cols(n_cols)
                    rows = utilities.labels_rows(n_rows)
                    # # with empty data and column labels but no row labels, should still be empty
                    save(pandas.DataFrame({},columns=cols),f.name)
                    with self.subTest(i=self.i_sub_test,msg=plate_size):
                        df_flat = plate_io.read_plate_dict(f.name)
                        assert len(df_flat["Sheet1"]) == 0
                    self.i_sub_test += 1
                    # # with empty data but column *and* row labels, should get all nans
                    all_nans = pandas.DataFrame({}, columns=cols,index=rows)
                    save(all_nans, f.name)
                    df_flat = plate_io.read_plate_dict(f.name)
                    with self.subTest(i=self.i_sub_test,msg=plate_size):
                        assert len(df_flat["Sheet1"]) == 1
                    self.i_sub_test += 1
                    with self.subTest(i=self.i_sub_test,msg=plate_size):
                        assert np.isnan(df_flat["Sheet1"][0].to_numpy()).all()
                    self.i_sub_test += 1
                    # # check that we can get data back properly, even if surrounded by nans
                    index_array = np.reshape(np.arange(0, (n_cols + 1) * (n_rows + 1)),
                                             (n_rows + 1, n_cols + 1)).astype(str)
                    index_array[1:, 0] = rows
                    index_array[0, 1:] = cols
                    nan_ones = np.ones_like(index_array,dtype=float) * np.nan
                    # surround the data in nans and make sure we can still get it
                    nans_and_data = np.concatenate([
                        np.concatenate([nan_ones, nan_ones, nan_ones], axis=1),
                        np.concatenate([nan_ones, index_array, nan_ones],axis=1),
                        np.concatenate([nan_ones, nan_ones, nan_ones], axis=1),
                    ])
                    save(pandas.DataFrame(nans_and_data), f.name)
                    df_flat = plate_io.read_plate_dict(f.name)
                    # make sure we recover the data
                    # (note that I ignore the first row and column of index_array
                    # and cat to float to better compare)
                    with self.subTest(i=self.i_sub_test,msg=plate_size):
                        assert len(df_flat["Sheet1"]) == 1
                    self.i_sub_test += 1
                    df_here = df_flat['Sheet1'][0]
                    with self.subTest(i=self.i_sub_test,msg=plate_size):
                        assert all(df_here.index == rows)
                        assert all(df_here.columns == cols)
                    self.i_sub_test += 1
                    just_data = index_array[1:,1:].astype(float)
                    with self.subTest(i=self.i_sub_test,msg=plate_size):
                        assert (df_here.to_numpy() == just_data).all()
                    self.i_sub_test += 1
                    # Make sure multiple sheets works
                    index_array_v2 = np.reshape(np.arange(0, (n_cols + 1) * (n_rows + 1)),
                                                (n_rows + 1, n_cols + 1)).astype(str)
                    index_array_v2 = index_array_v2[::-1]
                    index_array_v2[1:, 0] = rows
                    index_array_v2[0, 1:] = cols
                    just_data_v2 = index_array_v2[1:,1:].astype(float)
                    nans_and_data_v2 = np.concatenate([
                        np.concatenate([index_array, nan_ones, nan_ones], axis=1),
                        np.concatenate([nan_ones, index_array, index_array_v2],axis=1),
                        np.concatenate([index_array_v2, index_array_v2, index_array], axis=1),
                    ])
                    save(pandas.DataFrame(nans_and_data_v2), f.name)
                    df_flat_v2 = plate_io.read_plate_dict(f.name)
                    expected = [just_data,just_data,just_data_v2,just_data_v2,
                                just_data_v2,just_data]
                    with self.subTest(i=self.i_sub_test, msg=plate_size):
                        assert len(df_flat_v2['Sheet1']) == len(expected)
                    self.i_sub_test += 1
                    with self.subTest(i=self.i_sub_test, msg=plate_size):
                        for exp, s in zip(expected,df_flat_v2['Sheet1']):
                            assert (exp == s.to_numpy()).all()
                        self.i_sub_test += 1


    #pylint: disable=too-many-locals
    def test_99_basic_file_io(self):
        """
        test basic file io:

        (1) reading in csv and xlsx files
        (2) making sure the data is correct
        """
        self.i_sub_test = 0
        for k,video_k in self.resized_videos.items():
            all_matrices = [video_k[0,:,:,i] for i in range(3)]
            plate_df_colors = [ utilities.matrix_to_plate_df(m)
                                for m in all_matrices]
            all_concat = pandas.concat([p.reset_index() for p in plate_df_colors],
                                       axis=1)
            all_concat_reverse = pandas.concat([p.reset_index() for p in plate_df_colors][::-1],
                                               axis=1)
            all_concat_mult = [all_concat, all_concat_reverse,all_concat]
            mat_mult = all_matrices + all_matrices[::-1] + all_matrices
            # make a reversed version
            all_concat_mult_v2 = [all_concat_reverse,all_concat,all_concat_reverse]
            mat_mult_v2 =  all_matrices[::-1] + all_matrices + all_matrices[::-1]
            matrix = all_matrices[0]
            plate_df = plate_df_colors[0]
            # # first, test that just saving a single matrix works well
            for suffix,f_save in [[f"_{k}.csv",plate_df.to_csv],
                                  [f"_{k}.xlsx",plate_df.to_excel]]:
                with tempfile.NamedTemporaryFile(suffix=suffix) as f:
                    f_save(f.name)
                    self._check_plate_read_expected(f.name,
                                                    sheet_to_matrices={"Sheet1":[matrix]})
            # # next, check that saving multiple dataframes to the same file works
            for suffix in [f"_{k}.xlsx",f"_{k}.csv"]:
                with tempfile.NamedTemporaryFile(suffix=suffix) as f:
                    # saveing out to excel and csv are a littler different
                    save_all_plates(plate_df_colors=plate_df_colors,file_name=f.name)
                    self._check_plate_read_expected(f.name,
                                                    sheet_to_matrices={"Sheet1":all_matrices})
            # # Check that if we concatenate them column-wise, that also works
            for suffix,f_save in [[f"_{k}.csv",all_concat.to_csv],
                                  [f"_{k}.xlsx",all_concat.to_excel]]:
                with tempfile.NamedTemporaryFile(suffix=suffix) as f:
                    f_save(f.name)
                    self._check_plate_read_expected(f.name,
                                                    sheet_to_matrices={"Sheet1":all_matrices})
            # # Check that if we have multiple plates on rows and columns, that also works
            for suffix in [f"_{k}.xlsx",f"_{k}.csv"]:
                with tempfile.NamedTemporaryFile(suffix=suffix) as f:
                    # saveing out to excel and csv are a littler different
                    save_all_plates(plate_df_colors=all_concat_mult,file_name=f.name)
                    self._check_plate_read_expected(f.name,
                                                    sheet_to_matrices={"Sheet1":mat_mult})
            # # Multiple plates on rows and columns and multiple sheets, still works (just for XLSX)
            with tempfile.NamedTemporaryFile(suffix=f"_{k}.xlsx") as f:
                save_all_plates(plate_df_colors={"Sheet1":all_concat_mult,
                                                 "Sheet2":all_concat_mult_v2,
                                                 "Sheet3":all_concat_mult},
                                file_name=f.name)
                self._check_plate_read_expected(f.name,
                                                sheet_to_matrices={
                                                    "Sheet1": mat_mult,
                                                    "Sheet2":mat_mult_v2,
                                                    "Sheet3":mat_mult})



    def test_97_headers(self):
        """
        test that the header reading goes well
        """
        np.random.seed(42)
        self.i_sub_test = 0
        for save,suffix in [[lambda x,*args,**kw : x.to_excel(*args,**kw),".xlsx"],
                            [lambda x,*args,**kw: x.to_csv(*args,**kw),".csv"]]:
            with tempfile.NamedTemporaryFile(suffix=suffix) as f:
                for plate_size, plate_dict in utilities.plate_to_well_dict().items():
                    n_cols, n_rows = plate_dict["n_cols"], plate_dict["n_rows"]
                    cols = utilities.labels_cols(n_cols)
                    rows = utilities.labels_rows(n_rows)
                    index_array = np.reshape(np.arange(0, (n_cols + 1) * (n_rows + 1)),
                                             (n_rows + 1, n_cols + 1)).astype(str)
                    index_array[1:, 0] = rows
                    index_array[0, 1:] = cols
                    just_data_df = pandas.DataFrame(index_array[1:,1:],columns=cols)
                    just_data_df.index = rows
                    nan_ones = np.ones_like(index_array, dtype=float) * np.nan
                    # surround the data in nans and make sure we can still get it
                    nans_and_data = np.concatenate([
                        np.concatenate([nan_ones, index_array, nan_ones], axis=1),
                    ])
                    for n_rows in [1, 2, 3, 9, 0]:
                        n_cols_data = nans_and_data.shape[1]
                        header = randint(0, 100,size=(n_rows,n_cols_data))
                        # # try with just a single header
                        nans_and_data_and_header = \
                            np.concatenate([header,nans_and_data],axis=0)
                        save(pandas.DataFrame(nans_and_data_and_header), f.name,
                             index=False)
                        df_flat = plate_io.read_plate_dict(f.name, file_type="plate_and_header")
                        with self.subTest(i=self.i_sub_test,msg=f"{plate_size}"):
                            assert len(df_flat['Sheet1']) == 1
                        self.i_sub_test += 1
                        found_header, found_data = df_flat['Sheet1'][0]
                        with self.subTest(i=self.i_sub_test,msg=f"{plate_size}"):
                            found_header_float = found_header.to_numpy(dtype=float)
                            assert (found_header_float == header.astype(float)).all()
                        self.i_sub_test += 1
                        with self.subTest(i=self.i_sub_test,msg=f"{plate_size}"):
                            assert found_data.astype(float).equals(just_data_df.astype(float))
                        self.i_sub_test += 1
                        # # try with multiple different headers
                        header_v2 = randint(0, 100,size=(n_rows,n_cols_data))
                        header_v3 = randint(0, 100,size=(n_rows,n_cols_data))
                        nans_and_data_and_header_multi = \
                            np.concatenate([header,nans_and_data, # has header
                                            nans_and_data, # no header
                                            header_v2,nans_and_data, # v2 header
                                            header_v3,nans_and_data, # v3 header
                                            nans_and_data # no header
                                            ],axis=0)
                        save(pandas.DataFrame(nans_and_data_and_header_multi), f.name,
                             index=False)
                        df_flat_multi = plate_io.read_plate_dict(f.name,
                                                                 file_type="plate_and_header")
                        # second and last headers are empty
                        empty_header = np.ones(shape=(0,n_cols_data),dtype=float)
                        expected_header = [header,
                                           empty_header,
                                           header_v2,
                                           header_v3,
                                           empty_header]
                        with self.subTest(i=self.i_sub_test,msg=f"{plate_size}"):
                            assert len(df_flat_multi['Sheet1']) == len(expected_header)
                        self.i_sub_test += 1
                        for i,expected in enumerate(expected_header):
                            found_header, found_data = df_flat_multi['Sheet1'][i]
                            with self.subTest(i=self.i_sub_test,msg=f"{plate_size}"):
                                found_header_float = found_header.to_numpy(dtype=float)
                                assert (found_header_float == expected.astype(float)).all()
                            self.i_sub_test += 1
                            with self.subTest(i=self.i_sub_test,msg=f"{plate_size}"):
                                assert found_data.astype(float).equals(just_data_df.astype(float))
                            self.i_sub_test += 1

    def test_01_vendor_files(self):
        """
        Test reading in vendor files

        Can add these files using:

        find data/plate_examples/ -type f ! -name .DS_Store -exec git add {}
        """
        self.i_sub_test = 0
        in_dir = "data/plate_examples/input"
        in_dir_exp = "data/plate_examples/expected"
        example_files = [ os.path.join(in_dir,f)
                          for f in os.listdir(in_dir)]
        example_files_expected = [ os.path.join(in_dir_exp,os.path.basename(f))
                                   for f in example_files]
        for file_v,file_v_expected in zip(example_files,example_files_expected):
            plate_type = os.path.basename(file_v).split(".")[0]
            df_flat_v2 = plate_io.read_plate_dict(file_v, file_type=plate_type)
            # save out the file with this line:
            # save_all_plates(plate_df_colors=df_flat_v2, file_v_expected)
            n_expected = len(df_flat_v2["Sheet1"])
            n_rows =  len(df_flat_v2["Sheet1"][0])
            expected = read_plates_as_csv(file_v_expected, n_expected, n_rows)
            for s,s_expected in zip(df_flat_v2["Sheet1"],expected):
                with self.subTest(i=self.i_sub_test,msg=os.path.basename(file_v)):
                    assert all(s_expected == s)
                self.i_sub_test += 1

    def test_00_mario_as_xlsx(self):
        """
        make mario into an xlsx/csv and read back
        """
        self.i_sub_test = 0
        for video_mat,is_rgb in [[self.resized_videos_grey["3456"], False],
                                 [self.resized_videos["3456"], True]]:
            for suffix in [".csv",".xlsx"]:
                sheets = plate_io.matrix_to_video_dict(video_mat)
                with tempfile.NamedTemporaryFile(suffix=suffix) as f:
                    # save the plates out
                    plate_io.save_all_plates(sheets,file_name=f.name,index=True)
                    # first read the plates back in
                    time_rgb = plate_io.file_to_video(file_name=f.name,
                                                      file_type="DEFAULT PLATE",
                                                      is_rgb=is_rgb)
                    with self.subTest(i=self.i_sub_test):
                        if is_rgb:
                            # should matc exactly
                            assert (time_rgb == video_mat).all()
                        else:
                            # greyscale is floating point so should be close
                            np.testing.assert_allclose(time_rgb,
                                                       np.reshape(video_mat,time_rgb.shape))
                    # second, make sure the video works
                    self.i_sub_test += 1
                    kw_common = {'input_file':f.name,
                                 'file_type':"DEFAULT PLATE",
                                 'is_rgb':is_rgb,'fps':10}
                    with tempfile.NamedTemporaryFile(suffix=".gif") as f_out_gif:
                        with self.subTest(i=self.i_sub_test):
                            plate_tx.visualize_helper(output_file=f_out_gif.name,
                                                      **kw_common)
                        self.i_sub_test += 1
                    # make sure the png works
                    with tempfile.NamedTemporaryFile(suffix=".png") as f_out_png:
                        with self.subTest(i=self.i_sub_test):
                            plate_tx.visualize_helper(output_file=f_out_png.name,
                                                      **kw_common)
                        self.i_sub_test += 1



def read_plates_as_csv(file_v_expected,n_expected,n_rows):
    """

    :param file_v_expected: names of file
    :param n_expected:  how many plates we expect
    :param n_rows: how many rows per plate
    :return: list, length N, element is a single plate
    """
    df_expected = []
    i_offset = 0
    for _ in range(n_expected):
        df_here = pandas.read_csv(file_v_expected, index_col=0,
                                  nrows=n_rows, skiprows=i_offset)
        # +1 comes from the extra header row
        i_offset += n_rows + 1
        df_expected.append(df_here)
    return df_expected


if __name__ == '__main__':
    unittest.main()
