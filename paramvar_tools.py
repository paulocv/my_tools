
from toolbox.file_tools import remove_border_spaces, str_to_list


def read_csv_names(string):
    """Reads multiple strings separated by commas and removes border spaces.
    Example:
        "beta, pj ,  num_steps" --> ['beta', 'pj', 'num_steps']
    """
    return [remove_border_spaces(name) for name in string.split(',')]


def get_file_prefix(input_dict, std_prefix):
    """Gets the prefix string for the output files (including summary file).
    If file_prefix is not found in the input dict, a standard string is used.
    """
    try:
        file_prefix = input_dict["file_prefix"]
    except KeyError:
        file_prefix = std_prefix
        print("No file_prefix found on the input file. Using '{}' as prefix."
              "".format(file_prefix))
    return file_prefix


def get_varparams(input_dict, varparam_key="vary_parameters"):
    """
    Reads the "vary_parameters" input (i.e., the list of names of the
    parameters that must vary during all the simulations.
    Also interpret each variable parameter as a list.

    Returns a dictionary with the varying parameter names as keys and
    the corresponding lists of values as values.

    Parameters
    ----------
    input_dict : dict
        Dictionary with the inputs, duh.
    varparam_key : str
        Keyword for the varying parameter names. They should be found
        in the input_dict.

    Returns
    -------
    var_param_names
        List of the names of the varying parameters, in the order that
        they were written on the 'varparam_key' input from the input
        dict.
    values_list
        List of the varying parameters values lists, in the same order
        of var_param_names.
    """

    # Reads the names of the parameters that will vary
    try:
        var_param_names = read_csv_names(input_dict[varparam_key])
    except KeyError:
        raise KeyError("HEY, keyword '{}' was not found in"
                       "the input file.".format(varparam_key))

    # For each varying parameter, tries to interpret the corresponding list
    # of values in the input_dict.
    values_list = []
    for name in var_param_names:
        # Interprets the input as a list.
        values_list += [str_to_list(input_dict[name], name)]

    return var_param_names, values_list


def build_single_input_dict(mult_input_dict, keys_list, values_list):
    """Returns a single-input dict from a mult-input dict, for the given
    set of variable parameters.

    Parameters
    ----------
    mult_input_dict : dict
        Original dictionary with mult-inputs (sets of variable parameters).
    keys_list : list
        List of names of the variable parameters.
    values_list : list
        List of values of the variable parameters. Must follow the same order
        of the keys_list.
    """
    single_input_dict = mult_input_dict.copy()

    # For each variable parameter, replaces the entry on the mult-input,
    # transforming it into a single input dict.
    for i in range(len(keys_list)):
        single_input_dict[keys_list[i]] = values_list[i]

    return single_input_dict
