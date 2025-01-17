"""
All unit tests
"""
import unittest
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
        Placeholder
        """

if __name__ == '__main__':
    unittest.main()
