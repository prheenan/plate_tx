"""
Video-specific code
"""
from skvideo.io import vread
import cv2 as cv
import numpy as np
from tqdm import tqdm
from matplotlib import animation
import plot

def read_video(file_name):
    """

    :param file_name: file name
    :return: video as array
    """
    return vread(file_name,as_grey=False)


def resize_video(video_slice,px_width=48,px_height=32,
                 interpolation=cv.INTER_AREA):
    """

    :param video_slice: video to resize
    :param px_width: how wide
    :param px_height: how tall
    :param interpolation: what type of open interpolation flag (see cv.INTER_<X>
    https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html)

    :return: resized video
    """
    return np.array([ cv.resize(a_time,(px_width,px_height),0,0,
                      interpolation=interpolation)
                     for a_time in tqdm(video_slice)])

def save_comparison_video(frames,file_name,fps = 10,verbose=False):
    """

    :param frames: array like <T times, N width, M height, 3 RGB>
    :param file_name: output file
    :param fps: frames per second
    :param verbose: if True, prints status messaages
    :return: nothing, saves output
    """
    fig, _, ims = plot.flat_vs_matrix_figure(matrix=frames[0, :, :, :],
                                             animated=True)
    # Animation update function
    def update_figure(i):
        if verbose:
            print(f"Frame {i + 1:04d}/{len(frames):04d}")
        # Update the data
        ims[0].set_array(frames[i])
        ims[1].set_array(plot.flatten_rgb(frames[i]))
        return ims

    def init_func():
        return ims

    # Create the animation
    ani = animation.FuncAnimation(fig, update_figure, init_func=init_func,
                                  interval=1000 / fps, blit=True,
                                  frames=len(frames),
                                  cache_frame_data=False,
                                  repeat=True)

    ani.save(file_name, writer=animation.PillowWriter(fps=fps))
