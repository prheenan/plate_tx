"""
Plotting utilities
"""
from matplotlib import pyplot as plt
import numpy as np
import utilities

def plate_fig(plate_val,dpi=200,in_per_n=1/20,cmap=None):
    """
    Format a figure to display a platemap

    :param plate_val: plate
    :param dpi:dpi for figure
    :param in_per_n: inches per pixels
    :param cmap: color map
    :return:
    """
    n_rows, n_cols =  plate_val.shape[0], plate_val.shape[1]
    figsize = in_per_n *n_cols, in_per_n*n_cols
    fig = plt.figure(figsize=figsize,dpi=dpi)
    fig.tight_layout(pad=0)
    ax = plt.gca()
    format_for_plate(ax,n_rows=n_rows,n_cols=n_cols)
    ax.imshow(plate_val,cmap=cmap)
    return fig

def _default_if_none(val,default):
    """

    :param val: value
    :param default:  if none, return this instead
    :return:
    """
    return default if val is None else val

# pylint: disable=too-many-arguments,too-many-positional-arguments
def format_for_plate(ax=None, font_size=4, spacing_col=None, spacing_row=None,
                     n_rows=32, n_cols=48):
    """

    :param ax: axis
    :param font_size: font size
    :param spacing_col: show every <spacing_col> labels in columns
    :param spacing_row: show every <spacing_row> labels in columns
    :param n_rows: number of rows
    :param n_cols:  number of columns
    :return:  nothing, modifies axis
    """
    # determine how many labels to show
    if spacing_col is None or spacing_row is None:
        # then figure out the size based on the number of wells
        size = n_rows * n_cols
        size_defaults = [ [384,1],
                          [1536,2],
                          [3456,3]]
        for size_i,default in size_defaults:
            if size <= size_i:
                spacing_col = _default_if_none(spacing_col, default)
                spacing_row = _default_if_none(spacing_row, default)
                break
    if ax is None:
        ax = plt.gca()
    rows = utilities.labels_rows(n_rows)
    cols = utilities.labels_cols(n_cols)
    ax.set_xticks(ticks=list(range(0, n_cols))[::spacing_col],
                  labels=cols[::spacing_col],
                  font={"size": font_size})
    ax.tick_params(axis='both', which='both', length=0, pad=2)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_yticks(ticks=list(range(0, n_rows))[::spacing_row],
                  labels=rows[::spacing_row], font={"size": font_size})

def flatten_image(image):
    """
    flatten an RGB image into a single column

    :param image: RGB image, like <N,M,3>
    :return: RGB column, like <N*M,1,3>
    """
    if len(image.shape) == 3 and image.shape[-1] == 3:
        # RGB
        return np.reshape(image,(image.shape[0]*image.shape[1],1,3))
    # greyscale
    return np.reshape(image,(image.shape[0]*image.shape[1],1))

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def flat_vs_matrix_figure(matrix,flat=None,figsize=(4.25,2.1),dpi=200,
                          width_ratios=(10,1),font_size=5,animated=False):
    """
    compare flattened versu matriced figures

    :param matrix: matrix format of image
    :param flat: see flatten_image applied to matrix
    :param figsize: see plt.subplots
    :param dpi:see plt.subplots
    :param width_ratios:see plt.subplots
    :param font_size: of axis labels
    :param animated: if True, then set to animated
    :return: tuple of < figure, two axes, two image artists>
    """
    if flat is None:
        flat = flatten_image(matrix)
    # Create a figure and axes
    fig, axs = plt.subplots(nrows=1,ncols=2,dpi=dpi,layout="constrained",
                           figsize=figsize,width_ratios=width_ratios)
    ax = axs[0]
    format_for_plate(ax=ax,font_size=font_size,
                     n_cols=matrix.shape[1],n_rows=matrix.shape[0])
    # Create an initial image
    im_matrix = ax.imshow(matrix,interpolation='none',animated=animated)
    arrowprops={'arrowstyle':'simple,head_width=1,head_length=0.5,tail_width=0.5',
                'shrinkA': 0, 'shrinkB': 0,
                'lw':0.1,'facecolor':"k",'edgecolor':"k"}
    kw_arrow = {'xycoords':'figure fraction','arrowprops':arrowprops,
                'clip_on':False,'textcoords':'figure fraction','text':""}
    dx = 0.04
    offset_x = 0.805
    for y in [0.15, 0.45,0.8]:
        axs[0].annotate(xy=(offset_x+dx, y), xytext=(offset_x, y), **kw_arrow)
        axs[0].annotate(xy=(offset_x-dx, y), xytext=(offset_x, y), **kw_arrow)
    ax_pixels = axs[-1]
    ax_pixels.set_ylabel("Well (arb)")
    im_flat = ax_pixels.imshow(flat,aspect="auto",interpolation='none',animated=animated)
    ax_pixels.tick_params(axis='both', which='both',length=0,pad=2)
    ax_pixels.yaxis.set_label_position('right')
    ax_pixels.tick_params(top=False, labeltop=False, bottom=False,
                          labelbottom=False,labelleft=False)
    return fig, axs, [im_matrix,im_flat]
