"""
A module with functionalities to work with massive results from simulations,
typically with several varying parameters.
"""

import pandas as pd
import numpy as np
from toolbox.file_tools import read_file_header, read_config_strlist, read_csv_names, \
    str_to_list
from toolbox.paramvar_tools import get_varparams
# import matplotlib.pyplot as plt


def read_complete_output_file(filename,
                              entry_char=">", attribution_char="=", comment_char="#",
                              header_end="-----\n", varparams_key="vary_parameters",
                              delimiter='\t',
                              names=None):
    """Imports dataframe from an output summary file based on its HEADER.
    I.e., looks for the varying parameters in 'vary_parameters' entry, then
    interpret the lists of values.
    The DataFrame index is built as the cartesian product of all varying
    parameters.

    Inputs
    ------
    filename : str
        Name of the input file
    entry_char : str
        Character used to start an input line in the file header.
    attrbution_char : str
        Character that separates the key name from its value (header only).
    comment_char : str
        Character that indicates a comment. Used both for header and data.
    header_end : str
        String that defines the separation between the header and the main data.
        It must contain '\n' at the end.
    varparams_key : str
        Keyword for the varying parameter names list in the header. Ex:
        > vary_parameters = rho, kappa, pi, beta_ep
    delimiter : str
        horizontal delimiter between column entries
    names : list
        List of column names for the DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        Resulting data frame read from file.
    var_param_dict : dict
        Dictionary with the varying parameters names and their lists of values.
        This may be useful to unambiguous .loc indexing of the DataFrame rows,
        as the indexes are often floats.
    """

    # Reads the file header and interprets its valid entries (including varying
    #  parameters).
    file_header = read_file_header(filename, header_end)
    header_size = len(file_header) + 1
    input_dict = read_config_strlist(file_header, entry_char, attribution_char,
                                     comment_char)
    names_list, values_list = get_varparams(input_dict, varparams_key)

    # Constructs the MultiIndex object as the cartesian product of all varying
    # parameters
    index = pd.MultiIndex.from_product(values_list,
                                       names=names_list)

    # Reads the actual numerical data from file, interpreting it as a
    # pandas DataFrame with a multi-index.
    df = pd.read_table(filename, sep=delimiter,
                       skiprows=header_size, header=None,
                       comment=comment_char)

    # Eliminates a NAN column, which happens in case the line has another
    # delimiter at the end
    df = df.dropna(axis=1)

    # Sets the index and the useful columns
    df.index = index
    df = df.loc[:, len(values_list):]  # Removes the indexing columns

    # Sets the default value of the column 'names'
    # If the file header has an "outputs" topic, uses as column names
    # Else, uses integer indexes (starting from 0).
    if names is None:
        try:
            names = read_csv_names(input_dict["outputs"])
        except KeyError:
            names = list(range(len(df.columns)))

    df.columns = names  # Renames the columns

    var_param_dict = dict([(name, value)
                          for name, value in zip(names_list, values_list)])
    return df, var_param_dict


def read_mixed_output_file(filename, decimal_places=9,
                           entry_char=">", attribution_char="=", comment_char="#",
                           header_end="-----\n", varparams_names=None,
                           varparams_key="vary_parameters",
                           outputs_key="outputs", delimiter='\t',
                           names=None):
    """
    Reads an output summary file as a dataframe with a multiindex.
    Differently from 'read_complete_output_file', this function reads
    each parameter set from the first columns of the data, being more
    versatile (can be used with non-regularly distributed parameter
    sets.

    As float indexes may be used, the values are rounded to 'decimal_places',
    therefore numbers that differ bellow such precision will be considered
    as the same index values.

    Parameters
    ------
    filename : str
        Name of the input file
    decimal_places : int
        Number of decimal places bellow which floats are considered
        as equal indexes. This does not affect the data, only the index
        values.
    entry_char : str
        Character used to start an input line in the file header.
    attribution_char : str
        Character that separates the key name from its value (header only).
    comment_char : str
        Character that indicates a comment. Used both for header and data.
    header_end : str
        String that defines the separation between the header and the main data.
        It must contain '\n' at the end.
    varparams_names : list
        List of names of the parameters. If not informed, it is seached
        at the file header with the key specified as'varparams_key'.
    varparams_key : str
        Keyword for the varying parameter names list in the header. Ex:
        > vary_parameters = rho, kappa, pi, beta_ep
    outputs_key : str
        Keyword for the output topics, read from the file header.
    delimiter : str
        horizontal delimiter between column entries
    names : list
        List of column names for the DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        Resulting data frame read from file.
    """

    # Reads the file header and interprets its valid entries (including varying
    #  parameters).
    file_header = read_file_header(filename, header_end)
    header_size = len(file_header) + 1
    input_dict = read_config_strlist(file_header, entry_char, attribution_char,
                                     comment_char)

    # Tries to read the varying parameter names, if they are not informed
    # as 'varparams_names'
    if varparams_names is None:
        try:
            varparams_names = read_csv_names(input_dict[varparams_key])
        except KeyError:
            raise KeyError("Hey, keyword '{}' was not found in"
                           "the input file. If the file does not inform"
                           " the vary parameter names, this can be "
                           "informed to function {} as 'varparams_names'."
                           "".format(varparams_key, "read_mixed_output_file"))

    # Reads the database from file and removes inexistent entries
    df = pd.read_table(filename, sep=delimiter,
                       skiprows=header_size, header=None,
                       comment=comment_char)
    df = df.dropna(axis=1)

    # Separates the dataframe between index (parameter values) and actual data.
    indexing_df = df.loc[:, :len(varparams_names)-1]  # Index (parameters) columns
    values_df = df.loc[:, len(varparams_names):]  # Values (data) column

    # Sets the default value of the column 'names'.
    # If the file header has an "outputs" topic, uses as column names
    # Else, uses integer indexes (starting from 0).
    if names is None:
        try:
            names = read_csv_names(input_dict[outputs_key])
        except KeyError:
            names = list(range(len(values_df.columns)))

    # Renames the columns of the index and values dataframes
    indexing_df.columns = varparams_names
    values_df.columns = names

    # Index construction (by rounding the values)
    df_size = len(indexing_df)
    index = np.empty((len(varparams_names), df_size))

    # For each line of the index df
    for i in range(df_size):
        line = indexing_df.loc[i]
        # For each parameter value of the line
        for j, value in enumerate(line):
            # Stores the rounded values of the parameter values
            index[j][i] = round(value, decimal_places)

    # Converts the arrays into Pandas MultiIndex object and sets to df
    values_df.index = pd.MultiIndex.from_arrays(index, names=varparams_names)

    return values_df


def xs_by_dict(df, param_dict, column_names=None):
    """
    Makes a cross section of a pandas multiindex dataframe by
    taking a dictionary of the parameters (index) names and their
    values.

    Parameters
    ----------
    df : pandas.DataFrame
        Multi-index data frame.
    param_dict : dict
        Dictionary with the level names (as keys) and their values to
        be accessed (as values).
    column_names : str, list
        Column name to be used. If not informed, all columns are used.
    """
    if column_names is None:
        return df.xs(tuple(param_dict.values()),
                     level=tuple(param_dict.keys()))
    else:
        return df.xs(tuple(param_dict.values()),
                     level=tuple(param_dict.keys()))[column_names]
