"""Tools for plot styling - creation of style sheets."""

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import os
from functools import reduce
# import sys
from toolbox.file_tools import SEP, write_config_string, list_to_csv


STD_USETEX = False


# A nice style for journal plots, with large fonts, minor ticks,
# a latex font much better than default, and others.
# Based on a style from Luiz Alves.
mystyle_01 = {
    # Latex
    "text.usetex": STD_USETEX,
    "text.latex.preamble": [r"\usepackage[T1]{fontenc}",
                            r"\usepackage{lmodern}",
                            r"\usepackage{amsmath}",
                            r"\usepackage{mathptmx}"
                            ],
    # Axes configuration
    "axes.labelsize": 30,
    "axes.titlesize": 30,
    "ytick.right": "on",  # Right and top axis included
    "xtick.top": "on",
    "xtick.labelsize": "25",
    "ytick.labelsize": "25",
    "axes.linewidth": 1.8,
    "xtick.major.width": 1.8,
    "xtick.minor.width": 1.8,
    "xtick.major.size": 14,
    "xtick.minor.size": 7,
    "xtick.major.pad": 10,
    "xtick.minor.pad": 10,
    "ytick.major.width": 1.8,
    "ytick.minor.width": 1.8,
    "ytick.major.size": 14,
    "ytick.minor.size": 7,
    "ytick.major.pad": 10,
    "ytick.minor.pad": 10,
    "axes.labelpad": 15,
    "axes.titlepad": 15,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,  # Includes minor ticks
    "ytick.minor.visible": True,

    # Lines and markers
    "lines.linewidth": 4,
    "lines.markersize": 9,
    "lines.markeredgecolor": "k",  # Includes marker edge
    'errorbar.capsize': 4.0,

    # Legend
    "legend.numpoints": 2,  # Uses two points as a sample
    "legend.fontsize": 20,
    "legend.framealpha": 0.75,

    # Font
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica"],
    "font.size": 20,
    "mathtext.fontset": "cm",  # Corrects the horrible sans font for latex

    # Figure
    "figure.figsize": (9.1, 7.),
}

mystyle_01_docs = "A nice style for journal plots, with large fonts, minor ticks,"
mystyle_01_docs += "a latex font much better than default, and others."
mystyle_01_docs += "Based on a style from Luiz Alves."


def create_mpl_style(name, style_dict, convert_lists=True, docstring=None):
    """Creates a .mplstyle file in the matplotlib styles folder.

    If using from ipython, you have to restart the kernel for the changes to have effect.
    Then load the style with plt.style.use(name).

    Parameters
    ----------
    name : str
        Name of the style and its file name.
    style_dict : dict
        Dictionary with matplotlib rcParams.
    convert_lists : bool
        Converts lists and tuples to csv, so they are understood by matplotlib.
        Default is True.
    docstring : str
        A comment about the style.
    """

    # Gets the matplotlib config dir and possibly creates the style subdir
    mpl_style_dir = mpl.get_configdir() + SEP + "stylelib"
    if not os.path.exists(mpl_style_dir):
        os.system("mkdir {}".format(mpl_style_dir))

    # Makes some modifications, but preserves the original dict
    style_dict = style_dict.copy()

    # Converts lists to csv, which is how mpl understands
    if convert_lists:
        for key, val in style_dict.items():
            if type(val) in [list, tuple]:
                style_dict[key] = list_to_csv(val)

    # Exports the dict to file
    style_text = write_config_string(style_dict, entry_char="", attribution_char=":")
    with open(mpl_style_dir + SEP + name + ".mplstyle", "w") as fp:
        if docstring is not None:
            fp.write("# " + docstring + "\n")
        fp.write(style_text)


# ------------------------
# Figure size easy handler function
# ------------------------

def stdfigsize(scale=1, nx=1, ny=1, ratio=1.3):
    """
    Returns a tuple to be used as figure size.
    -------
    returns (7*ratio*scale*nx, 7.*scale*ny)
    By default: ratio=1.3
    If ratio<0 them ratio = golden ratio
    """
    if ratio < 0:
        ratio = 1.61803398875

    return 7 * ratio * scale * nx, 7 * scale * ny

# ---------------------------
# Color, linestyle and other sequences
# ---------------------------


# Aux function: least common multiple.
def _lcm(x, y):
    """Least common multiple via brute (incremental) search."""
    if x > y:
        z = x
    else:
        z = y

    while True:
        if(z % x == 0) and (z % y == 0):
            lcm = z
            break
        z += 1

    return lcm


# Qualitative printer friendly only color seqs
colorbrewer_pf_01 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
                     '#a65628', '#f781bf', '#999999']
colorbrewer_pf_02 = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
colorbrewer_pf_03 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
                     '#b3de69', '#fccde5']

# Colorblind friendly only color seqs
colorbrewer_cbf_01 = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']
colorbrewer_cbf_02 = ['#1b9e77', '#d95f02', '#7570b3']
colorbrewer_cbf_03 = ['#66c2a5', '#fc8d62', '#8da0cb']


def set_color_cycle(colors):
    """Warning: this will only setup the colors, reseting all other cyclic
    properties.

    To set up a property without affecting the other, implement
    change_prop_cycle function.
    """
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)


def set_composite_prop_cycle(**props):
    """Sets the prop cycle to combinations of properties from seqs that
    are not necessarily commensurable.
    Each property is advanced at each new plot, unlike
    set_product_prop_cycle.

    Arguments are keywords as prop_name=prop_list.
    """
    # lcm()?? # Current function is for 2 values only.

    # Gets the multiplication factor for each prop sequence
    lens = [len(vals) for vals in props.values()]
    total_len = reduce((lambda x, y: x * y), lens)  # Product
    mult_facs = [total_len // l for l in lens]

    # Creates the composite cycler
    comp_props = {key: n*list(vals)
                  for n, (key, vals) in zip(mult_facs, props.items())}
    prop_cycle = cycler(**comp_props)

    # Sets it to mpl and returns
    plt.rcParams["axes.prop_cycle"] = prop_cycle
    return prop_cycle


def set_product_prop_cycle(**props):
    """Sets the prop cycle to combinations of properties from seqs that
    are not necessarily commensurable.
    The properties defined at last will cycle first ("faster"), followed
    by the next and so on.

    Arguments are keywords as prop_name=prop_list.
    """
    # Composite prop cycler
    cycle_list = []
    for key, vals in props.items():
        cycle_list.append(cycler(**{key: vals}))

    prop_cycle = reduce((lambda x, y: x * y), cycle_list)  # Product

    # Sets it to mpl and returns
    plt.rcParams["axes.prop_cycle"] = prop_cycle
    return prop_cycle