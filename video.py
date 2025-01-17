"""
Video-specific code
"""
import os
from skvideo.io import vread
import cv2 as cv
import numpy as np
from tqdm import tqdm
from matplotlib import animation
import plot

def read_demo_video(start_px=50,stop_px=750,**kw):
    """

    :param kw: passed to read_video
    :param start_px: where to start reading in height
    :param stop_px: where to stop reading in height
    :return: see  read_video, except the demo video
    """
    # crop out the black bands
    demo_path = os.path.join(os.path.dirname(__file__),"data/mario-trim.mov")
    return read_video(demo_path,**kw)[:,start_px:stop_px,:,:]

def read_video(file_name,as_grey=False,**kw):
    """

    :param file_name: file name
    :param as_grey: if true, read as greyscale
    :param kw: passed directly to read_video
    :return: video as ndarray of dimension (T, M, N, C), where
        T is the number of frames,
        M is the height,
        N is width
        C is depth (e.g., C=3 for RGB, C=1 for greyscale)
    """
    return vread(file_name,as_grey=as_grey,**kw)


def resize_video(video_slice,px_width=48,px_height=32,
                 interpolation=cv.INTER_AREA,disable_tqdm=False):
    """

    :param video_slice: video to resize
    :param px_width: how wide
    :param px_height: how tall
    :param interpolation: what type of open interpolation flag (see cv.INTER_<X>
    :param disable_tqdm: if true, disable tqdm
    https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html)

    :return: resized video
    """
    return np.array([ cv.resize(a_time,(px_width,px_height),0,0,
                      interpolation=interpolation)
                     for a_time in tqdm(video_slice,disable=disable_tqdm)])

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
