from toolbox.file_tools import remove_border_spaces, str_to_list


def read_csv_names(string):
    """Reads multiple strings separated by commas and removes border spaces.
    Example:
        "beta, pj ,  num_steps" --> ['beta', 'pj', 'num_steps']
    """
    return [remove_border_spaces(name) for name in string.split(',')]


def read_sequence_of_tuples(string):
    """Reads a sequence of tuples from a string, returning indeed a
    list of tuples.

    Examples:
    "(beta2, mu2), (a, b, c)" --> [("beta2", "mu2), ("a", "b", "c")]
    "( )"  -->  []
    " "  -->  ValueError
    " (a, b " -->  ValueError

    Parameters
    ----------
    string : str
    """
    result = []

    # Finds all opening ( symbols
    # If none is found, raises an error
    str_split = string.split("(")[1:]  # Eliminates the first, which is spurious
    if len(str_split) == 0:
        raise ValueError("Hey, check the parentheses at '{}'"
                         "".format(string))

    # For each substring after a (, finds the next ) and registers the names
    # in the middle to a tuple.
    for substring in str_split:
        # Finds the closing ) symbol for the current set
        index = substring.find(")")
        if index == -1:  # ")" not found
            raise ValueError("Hey, check the parentheses at '{}'"
                             "".format(string))

        # Includes the entry by reading from csv to list, then to tuple.
        result.append(tuple(read_csv_names(substring[:index])))

    # For consistency, checks if there aren't missing opening (
    # This avoids silent errors.
    if len(result) != len(string.split(")")) - 1:
        raise ValueError("Hey, check the parentheses at '{}'"
                         "".format(string))

    # Manually returns an empty list if there's a single empty entry
    if len(result) == 1 and result[0] == ("",):
        return []

    return result


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


def get_varparams_nozip(input_dict, varparam_key="vary_parameters"):
    """
    Reads the "vary_parameters" input (i.e., the list of names of the
    parameters that must vary during all the simulations.)
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
        raise KeyError("Hey, keyword '{}' was not found in "
                       "the input file.".format(varparam_key))

    # For each varying parameter, tries to interpret the corresponding list
    # of values in the input_dict.
    values_list = []

    # If no parameters to vary were informed, the returned list must be
    # manually set to empty
    if var_param_names[0] == "" and len(var_param_names) == 1:
        var_param_names = []
        print("Warning: no parameters to vary.")

    for name in var_param_names:
        # Interprets the input as a list.
        values_list += [str_to_list(input_dict[name], name)]

    return var_param_names, values_list


def get_varparams_with_zip(input_dict, varparam_key="vary_parameters",
                           zip_key="zip_parameters"):
    """
    Reads the "vary_parameters" input (i.e., the list of names of the
    parameters that must vary during all the simulations.) and the
    "zip_parameters input (i.e., the sets of parameters that should
    be varied together).

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
    zip_key : str
        Keyword for the zip parameter name sets. If not found at input_
        _dict, returns the same as get_varparams.
    # check_list_sizes : bool
        NOT IMPLEMENTED PARAMETER. Always False, thus silently accepts.
        If True, the parameters that are zipped together must have a
        list with the same size, and an error is raised if a length is
        different. If False, silently allows different sizes, truncating
        by the smallest one.

    Returns
    -------
    var_param_names
        List of the names of the varying parameters, in the order that
        they were written on the 'varparam_key' input from the input
        dict, and with tuples for the zipped parameters. Each zipped tuple
        replaces the original position of its "head" parameter, which is the first
        element of the tuple.
    values_list
        List of the varying parameters values lists, following the same
        order convention as var_param_names.
    """
    # Regularly reads the list of varying parameters and their values.
    var_param_names, values_list = get_varparams(input_dict, varparam_key)

    # Reads gets the zip parameter entry from input dict
    # If not found, simply returns the results without any zip
    try:
        zip_names_str = input_dict[zip_key]
    except KeyError:
        # If no zip_params keyword is found, simply returns the regular
        # parameter list.
        return var_param_names, values_list

    # Gets the tuples of names that will be zipped together
    zip_param_names = read_sequence_of_tuples(zip_names_str)

    # Reshapes the lists of names and values to include the zipped params
    for names in zip_param_names:  # For each set of zipped params
        # Finds the index of each name in the regular (flat) list
        # The list.index() method raises an error if not found.
        i_names = []
        for name in names:
            i_names.append(var_param_names.index(name))

        # The first name of the tuple (head) is replaced by the tuple itself,
        # and the zipped values
        var_param_names[i_names[0]] = names
        values_list[i_names[0]] = list(zip(*[values_list[i] for i in i_names]))

        # The other parameters from the tuple are simply removed from their
        # original positions. Uses list comprehension to simplify code.
        # Alternative implementation calls a sort method and removes
        # in reverse order.
        size = len(var_param_names)
        var_param_names = [var_param_names[i] for i in range(size) if i not in i_names[1:]]
        values_list = [values_list[i] for i in range(size) if i not in i_names[1:]]

    return var_param_names, values_list


# Aias to the get_varparams that supports parameter zipping.
get_varparams = get_varparams_with_zip


def _to_tuple(arg):
    # This function avoids a nonsense warning from PyCharm
    return tuple(arg)


def ziplist_to_flat(zipped_list, return_type=_to_tuple):
    """For a list that possibly contains tuples of zipped values,
    returns a flattened copy in which parameters inside
    the tuples are brought back to the first level, at the order that
    they appear.

    By default, the returned object is converted to tuple, which is
    more convenient for paramvar.

    Notice: regular values from the names_list cannot be tuples, as
    they will be confused with a zipped list of values.
    """
    flat_list = []
    for element in zipped_list:

        if type(element) is tuple:
            flat_list += list(element)  # Simply appends the values inside the tuple

        else:
            flat_list.append(element)

    return return_type(flat_list)


def zip_params_to_flat(names_list, values_list):
    """
    CURRENTLY NOT IN USE
    For the lists of parameter names and their values,
    that possibly contains zipped parameters (in tuples),
    returns a flattened copy in which parameters inside
    the tuples are brought back to the first level, at the order that
    they appear.

    Similar to ziplist_to_flat, but makes the job in names and values
    list simultaneously.

    Notice: regular values from the names_list cannot be tuples, as
    they will be confused with a zipped list of values.
    """
    flat_names = []
    flat_values = []

    for name, value in zip(names_list, values_list):

        # Detects zipped parameters if the name is a tuple instead of str
        if type(name) is tuple:
            flat_names += list(name)  # Simply appends the values inside the tuple
            flat_values += list(value)

        else:
            flat_names.append(name)
            flat_values.append(value)

    return flat_names, flat_values


def build_single_input_dict(mult_input_dict, keys_list, values_list):
    """Returns a single-input dict from a mult-input dict, for the given
    set of variable parameters.

    Parameters
    ----------
    mult_input_dict : dict
        Original dictionary with mult-inputs (sets of variable parameters).
    keys_list : list
        List of names of the variable parameters.
    values_list : list or tuple
        List of values of the variable parameters. Must follow the same order
        of the keys_list.
    """
    single_input_dict = mult_input_dict.copy()

    # For each variable parameter, replaces the entry on the mult-input,
    # transforming it into a single input dict.
    for i in range(len(keys_list)):
        single_input_dict[keys_list[i]] = values_list[i]

    return single_input_dict
