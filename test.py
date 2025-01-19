"""
All unit tests
"""
import unittest
import tempfile
import pandas
import numpy as np
import video
import utilities
import plate_io

class MyTestCase(unittest.TestCase):
    """
    Tester class
    """
    def __init__(self,*args,**kwargs):
        """
        Intialization (blank)
        """
        super().__init__(*args,**kwargs)
        self.i_sub_test = 0

    @classmethod
    def setUpClass(cls):
        """

        :return: Nothing, sets up the class
        """
        # read in the video
        cls.video_matrix = video.read_video("data/mario-trim.mov",
                                            num_frames=10)
        # https://en.wikipedia.org/wiki/Microplate#Formats_and_Standardization_efforts
        cls.resized_videos = {}
        for plate,plate_dict in utilities.plate_to_well_dict().items():
            n_rows,n_cols = plate_dict["n_rows"],plate_dict["n_cols"]
            cls.resized_videos[plate] = \
                video.resize_video(cls.video_matrix,px_width=n_cols,px_height=n_rows,
                                   disable_tqdm=True)

    def _check_plate_read_expected(self,file_name,sheet_to_matrices):
        """

        :param file_name: file name saved
        :param sheet_to_matrices: expectation: key is sheet name, value
        is list of matrices expected in order
        """
        sheet_to_dfs = plate_io.plate_to_flat(file_name)
        for sheet_name, matrices in sheet_to_matrices.items():
            with self.subTest(i=self.i_sub_test, msg=file_name):
                # should only have one file output
                assert len(sheet_to_dfs[sheet_name]) == len(matrices)
            self.i_sub_test += 1
            for i,m in enumerate(matrices):
                with self.subTest(i=self.i_sub_test, msg=file_name):
                    # make sure the data we read back in is correct
                    assert (sheet_to_dfs[sheet_name][i].to_numpy() == m).all()
                self.i_sub_test += 1

    def test_01_basic_file_io(self):
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
                    if suffix.endswith(".csv"):
                        plate_df_colors[0].to_csv(f.name)
                        for p in plate_df_colors[1:]:
                            p.to_csv(f.name,mode="a")
                    else:
                        plate_df_colors[0].to_excel(f.name)
                        i_row = len(plate_df_colors[0])+1
                        for p in plate_df_colors[1:]:
                            # pylint: disable=abstract-class-instantiated
                            with pandas.ExcelWriter(f.name, mode="a",engine="openpyxl",
                                                    if_sheet_exists='overlay') as xlsx:
                                p.to_excel(xlsx,startrow=i_row)
                                i_row += len(p) + 1
                    self._check_plate_read_expected(f.name,
                                                    sheet_to_matrices={"Sheet1":all_matrices})
            # # Check that if we concatenate them column-wise, that also works
            all_concat = pandas.concat([p.reset_index() for p in plate_df_colors], axis=1)
            for suffix,f_save in [[f"_{k}.csv",all_concat.to_csv],
                                  [f"_{k}.xlsx",all_concat.to_excel]]:
                with tempfile.NamedTemporaryFile(suffix=suffix) as f:
                    f_save(f.name)
                    self._check_plate_read_expected(f.name,
                                                    sheet_to_matrices={"Sheet1":all_matrices})

    def test_00_conversions(self):
        """
        Test converting from matrices to
        """
        self.i_sub_test = 0
        for k,video_k in self.resized_videos.items():
            # just look at the first time point and red channel
            matrix = video_k[0, :, :, 0]
            plate_df = utilities.matrix_to_plate_df(matrix)
            flat_df = utilities.plate_to_flat_df(plate_df)
            plate_df_2 = utilities.flat_to_plate_df(flat_df)
            flat_df_2 = utilities.plate_to_flat_df(plate_df_2)
            matrix_2 = utilities.plate_df_to_matrix(plate_df_2)
            matrix_3 = utilities.flat_df_to_matrix(flat_df_2)
            # make sure the two matrics match
            with self.subTest(i=self.i_sub_test,msg=k):
                np.testing.assert_allclose(matrix_2,matrix)
            self.i_sub_test += 1
            with self.subTest(i=self.i_sub_test,msg=k):
                np.testing.assert_allclose(matrix_3,matrix)
            self.i_sub_test += 1
            # make sure the plate translation matches
            with self.subTest(i=self.i_sub_test,msg=k):
                assert plate_df_2.equals(plate_df)
            self.i_sub_test += 1
            # make sure the flat translation matches
            with self.subTest(i=self.i_sub_test,msg=k):
                assert flat_df_2.equals(flat_df)
            self.i_sub_test += 1

if __name__ == '__main__':
    unittest.main()
