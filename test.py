"""
All unit tests
"""
import unittest
import numpy as np
import video
import utilities

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

    def test_conversions(self):
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
