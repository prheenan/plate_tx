"""
CLI interepreter
"""
import click
from click import ParamType
import matplotlib
from matplotlib import pyplot as plt
import plate_io
import utilities
import plot
import video

class BoolType(ParamType):
    """
    Defines click boolean stlye argument
    """
    def __init__(self):
        """
        Initialization
        """

    def get_metavar(self, param):
        """

        :param param: name of parametr
        :return:  help string
        """
        return 'Choice([TRUE/True/FALSE/False])'


    def convert(self, value, _, __):
        """

        :param value: value to convert
        :param _: not used
        :param __: not used
        :return: boolean value
        """
        upper = str(value).upper()
        if upper == "TRUE":
            return True
        elif upper == "FALSE":
            return False
        self.fail(f"Invalid value: {value}. Expected TRUE/FALSE")
        return False


@click.group()
def cli():
    """
    grouping function; left blank
    """

def _default_output_if_none(output_file,input_file):
    """

    :param output_file: output file.
    :param input_file: input file
    :return: output file if it exists, otherwise default output
    """
    if output_file is None:
        split = input_file.split(".")
        return ".".join(split[:-1]) + "_out_" + split[-1]
    # not None
    return output_file

def convert_helper(input_file,input_type,output_type,output_file):
    """
    :param input_file: to read from
    :param input_type:  file type to read
    :param output_type: either flat or plate
    :param output_file: where to output
    :return:
    """
    output_file = _default_output_if_none(output_file, input_file)
    dict_of_plates = plate_io.read_plate_dict(input_file, file_type=input_type)
    if output_type == "PLATE":
        dict_output = dict_of_plates
    elif output_type == "FLAT":
        dict_output = {k:[utilities.plate_to_flat_df(e)
                          for e in dict_of_plates[k]]
                       for k in dict_of_plates.items()}
    else:
        raise ValueError("Didn't understand")
    plate_io.save_all_plates(plate_df_colors=dict_output,
                             file_name=output_file)

#pylint : disable=too-many-positional-arguments,too-many-arguments
def visualize_helper(input_file,output_file,input_type="DEFAULT PLATE",
                     is_rgb=True,fps=10,cmap=None):
    """

    :param input_file: plate file
    :param output_file:  should be image file (e.e.g png) or gif (for video)
    :param input_type: of input file
    :param is_rgb:  if true, input file represent RGB data, each sheet is frame
    :param fps: frames per second
    :param cmap: color map for image
    :return:
    """
    output_file = _default_output_if_none(output_file, input_file)
    frames = plate_io.file_to_video(file_name=input_file, file_type=input_type,
                                    is_rgb=is_rgb)
    if output_file.endswith(".gif"):
        video.save_comparison_video(frames=frames, file_name=output_file,
                                    fps=fps, verbose=False)
    else:
        # assume image; use first plate
        if is_rgb:
            cmap = None
            matrix = frames[0, :, :, :]
        else:
            cmap = matplotlib.colormaps["plasma"] if cmap is None else cmap
            matrix = frames[0, :, :]
        fig = plot.plate_fig(plate_val=matrix,cmap=cmap)
        fig.savefig(output_file)
        plt.close(fig)


@cli.command()
@click.option('--input_file', required=True,
              type=click.Path(dir_okay=False,readable=True,exists=True))
@click.option('--output_file', required=False,default=None,
              type=click.Path(dir_okay=False,writable=True))
@click.option("--input_type",required=False,
              type=click.Choice(plate_io.PLATE_OPTIONS))
@click.option("--output_type",default="FLAT",required=False,
              type=click.Choice(["PLATE","FLAT"]))
def visualize(**kw):
    """

    :param kw: see visualize_helper
    :return: see visualize_helper
    """
    return visualize_helper(**kw)



@cli.command()
@click.option('--input_file', required=True,
              type=click.Path(dir_okay=False,readable=True,exists=True))
@click.option("--input_type",required=False,default="DEFAULT PLATE",
              type=click.Choice(plate_io.PLATE_OPTIONS))
@click.option('--output_file', required=False,default=None,
              type=click.Path(dir_okay=False,writable=True))
@click.option("--is_rgb",required=False,default=False,
              type=BoolType())
@click.option("--fps",required=False,default=10,
              type=int)
@click.option("--cmap",required=False,default=None,
              type=click.Choice(list(matplotlib.colormaps.keys())))
def convert(**kw):
    """

    :param kw: see convert_helper
    :return: see convert_helper
    """
    return convert_helper(**kw)


if __name__ == "__main__":
    cli()
