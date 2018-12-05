import os
import sys

SEP = "/"  # Use '/' for Linux and '\\' for Windows.
HEADER_END = "-----\n"

# --------------------------------
# FOLDER, FILE AND ARGV OPERATIONS
# --------------------------------


def file_overwrite_check(file_path):

    # Checks if the file already exists and prompt an action to the user.
    if os.path.exists(file_path):
        answer = input("\nWarning: file '{}' already exists, meaning that "
                       "you may be about to overwrite an older simulation.\n"
                       "Do you want to stop ('s'), rename ('r') or overwrite it"
                       "anyway ('o')?\n".format(file_path))
        # If the user wants to overwrite it anyway:
        if answer == "o":
            return file_path
        # If the user wants to rename the file (uses the same folder)
        elif answer == "r":
            file_path = os.path.dirname(file_path) + SEP
            file_path += input(file_path)
            return file_path
        # If the user wants to stop
        elif answer == "s":
            print("Quitting now...")
            quit()
        else:
            print("What did you type dude!?? I'll quit anyway, dumb...")
            quit()

    # If file does not exist, return its path anyway
    return file_path


def read_argv_optional(argi, dtype=None, default=None):
    """Reads an option from argv[argi]. Returns a default value
     if the argument is not present. Also converts to dtype otherwise.
    """
    try:
        if dtype is None:
            res = sys.argv[argi]
        else:
            res = dtype(sys.argv[argi])
    except IndexError:
        res = default

    return res


def get_folder_name_from_argv(argi=2, root_folder="", argi_check=True):
    """Reads a folder path from argv (argv[2] by standard).
    Adds the separator character (/, \\) if it was forgotten.
    Checks if the folder exists, and creates it otherwise.

    If the corresponding position in argv is not informed, asks for the
    user the path of the folder, starting from a given root folder.
    """
    # First tries to read the output folder name from argv[2]
    try:
        output_folder = sys.argv[argi]
    except IndexError:
        if argi_check:
            # If argv[argi] was not passed, asks the user for the output folder.
            output_folder = root_folder
            output_folder += input("Output folder path was not informed. Please inform:\n"
                                   "{}".format(root_folder))
        else:
            raise IOError("Hey, argv[{}] was not found!".format(argi))

    # Adds the SEP (/ or \\) character to the end of the folder name.
    if output_folder[-len(SEP):] != SEP:
        output_folder += SEP

    # Checks if the folder does not exist. Creates it, in this case.
    if not os.path.exists(output_folder):
        os.system("mkdir -p '{}'".format(output_folder))

    return output_folder


def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.system("mkdir -p '{}'".format(folder_path))
    else:
        print("Folder '{}' already exists.".format(folder_path))

# --------------------------------------------------------------------
# USEFUL STRING OPERATIONS
# --------------------------------------------------------------------


def remove_border_spaces(string):
    """ Strips the whitespace borders of a string.
    Used inside 'read_config_file' function.
    """
    if type(string) != str:
        raise TypeError(
            "Hey, a non-string object was passed to function "
            "'Remove_border_spaces'!")

    if string == '':
        return string

    while string[0] == ' ' or string[0] == '\t':
        string = string[1:]
        if string == '':
            return string

    while string[-1] == ' ' or string[-1] == '\t':
        string = string[:-1]
        if string == '':
            return string

    return string


def str_to_list(string, key_name=""):
    """Evaluates a string as a list. Checks if border characters are
    '[' and ']', to avoid bad typing.

    key_name : string (optional)
        Name of the parameter. Useful for error message.
    """
    if string[0] == '[' and string[-1] == ']':
        return list(eval(string))
    else:
        raise ValueError("Hey, bad parameter or list of parameters"
                         " in {} = {}".format(key_name,
                                              string))


def str_to_bool(string, truelist=None):
    """Returns a boolean according to a string. True if the string
    belongs to 'truelist', false otherwise.
    By default, truelist has "True" only.
    """
    if truelist is None:
        truelist = ["True"]
    return string in truelist


def get_bool_from_dict(input_dict, key, truelist=None, raise_keyerror=False):
    """Returns a boolean read from a string at an input dictionary.
    True if the string belongs to 'truelist', false otherwise.
    By default, truelist has "True" only.

    Parameters
    ----------
    input_dict : dict
        Input dictionary.
    key : any hashable
        Keyword for the boolean to be read.
    truelist : list
        List of strings that are considered as True value.
    raise_keyerror : bool
        Defines if a key error is raised if the required key is not
        found on input_dict. Default is False, meaning no key error.
        In this case, the function returns False.
    """
    if truelist is None:
        truelist = ["True"]

    try:
        return input_dict[key] in truelist
    except KeyError:
        if raise_keyerror:
            raise KeyError("Hey, key '{}' was not found on input dict."
                           "".format(key))
        else:
            return False


def seconds_to_hhmmss(time_s):
    """Converts a time interval given in seconds to hh:mm:ss.
    Input can either be a string or floating point.
    'ss' is rounded to the closest integer.

    """
    time_s = float(time_s)

    hh = int(time_s/3600)
    time_s = time_s % 3600
    mm = int(time_s/60)
    ss = round(time_s % 60)

    return "{}h{}m{}s".format(hh, mm, ss)


def list_to_csv(parlist):
    """Returns a csv string with elements from a list."""
    result_str = ""
    for par in parlist:
        result_str += str(par) + ", "
    return result_str[:-2]


def read_csv_names(string):
    """Reads multiple strings separated by commas and removes border spaces.
    Example:
        "beta, pj ,  num_steps" --> ['beta', 'pj', 'num_steps']
    """
    return [remove_border_spaces(name) for name in string.split(',')]

# -------------------------------------------------------------------
# CONFIGURATION FILE common operations
# -------------------------------------------------------------------


def read_optional_from_dict(input_dict, key, standard_val=None,
                            typecast=None):
    """Tries to read an option from a dictionary. If not found, a
    standard value is returned instead. If no standard value is
    informed, key error is raised. Data can also be converted by
    a type cast.
    The given standard value is not converted by typecast.
    """
    try:
        val = input_dict[key]
    except KeyError:
        if standard_val is None:
            raise KeyError("Hey, parameter '{}' was not found on dict."
                           "".format(key))
        else:
            return standard_val
    # Data conversion.
    if typecast is None:
        return val
    else:
        return typecast(val)


def read_config_file(file_path, entry_char='>', attribution_char='=',
                     comment_char='#', endline=None):
    """Function that reads 'markup' files.
     It opens the file and looks for lines with 'entry_char'. Example:
        > option_name: value  #This is a comment

    The ':' can be replaced by any combination of characters specified
    as 'attribution_char' keyword argument.

    Inputs
    ----------
    file_path : str
        Name of the file to be read.
    entry_char : str
        Character used to start an input line in the config file.
    attrbution_char : str
        Character that separates the key name from its value.
    comment_char : str
        Character that indicates a commentary. Everything after this
        character is ignored.
    endline : str
        Termination line string. If this line is found on the file,
        the reading is terminated and the function returns the results
        gathered until this point.

    Returns
    ----------
    result_dictio : dict
        Dictionary with all the options read from file.
    """
    # File opening and storing
    fp = open(file_path, 'r')
    file_strlist = fp.read().splitlines()
    fp.close()

    return read_config_strlist(file_strlist, entry_char, attribution_char,
                               comment_char, endline)


def read_config_strlist(strlist, entry_char='>', attribution_char='=',
                        comment_char='#', endline=None):
    """ Similar to 'read_config_file', but works at a string or a list
    of strings instead.
    Function that reads 'markup' content from a list of strings.
    It looks for lines with 'entry_char'. Example:
        > option_name = value  #This is a comment

    The '=' can be replaced by any combination of characters specified
    as 'attribution_char' keyword argument.

    Inputs
    ----------
    strlist : list, str
        String list to be read. Optionally, if a single string is passed,
        it is transformed into a string list by spliting on '\n' chars.
    entry_char : str
        Character used to start an input line in the config file.
    attrbution_char : str
        Character that separates the key name from its value.
    comment_char : str
        Character that indicates a commentary. Everything after this
        character is ignored.
    endline : str
        Termination line string. If this line is found on the string,
        the reading is terminated and the function returns the results
        gathered until this point.
        Obs: may or not contain the '\n' character at the end. Both
        cases work.

    Returns
    ----------
    result_dictio : dict
        Dictionary with all the options read from file.
    """

    # Checks if the input is a single string, and then splits it by \n.
    if type(strlist) == str:
        strlist = strlist.splitlines()

    # Size of attribution and entry chars (strings)
    entry_char_len = len(entry_char)
    attr_char_len = len(attribution_char)

    # Removes the '\n' at the end of 'endline'
    if endline is not None and len(endline) != 0:
        if endline[-1] == '\n':
            endline = endline[:-1]

    # Main loop over the list of strings
    result_dictio = {}
    for line in strlist:

        # Stops the loop if an 'endline' is found.
        if line == endline:
            break

        # Gets only lines which have the entry character at the start
        if line[0:entry_char_len] != entry_char:
            continue

        # Line text processing
        # Ignores everything after a comment character
        line = line.split(comment_char)[0]
        # Eliminates the initial (entry) character
        line = line[entry_char_len:]

        # Separation between key and value
        # Finds where is the attribution char, which separates key from
        # value.
        attr_index = line.find(attribution_char)
        # If no attribution char is found, raises an exception.
        if attr_index == -1:
            raise ValueError(
                "Heyy, the attribution character '" + attribution_char +
                "' was not found in line: '" + line + "'")
        key = remove_border_spaces(line[:attr_index])
        value = remove_border_spaces(line[attr_index + attr_char_len:])

        # Finally adds the entry to the dictionary
        result_dictio[key] = value

    return result_dictio


def entry_string(key, value, entry_char=">", attribution_char="=",
                 end_char="\n"):
    """Converts a keyword and a value to an accepted input string for
    'read_config_file.

    Inputs
    ----------
    key : str
        Keyword (name of the option/parameter). If not string, a
        conversion is attempted.
    value : str
        Value of the option/parameter. If not a string, a conversion
        is attempted.
    entry_char : str
        Character to start the line.
    attribution_char : str
        Character that separates the key from the value.
    end_char : str
        Character inserted at the end of the string.

    Returns
    ----------
    result_str : str
        String with an input line containing '> key = value'.
    """
    result_str = entry_char + " "
    result_str += str(key)
    result_str += " " + attribution_char + " "
    result_str += str(value)
    result_str += end_char
    return result_str


def write_config_string(input_dict, entry_char='>', attribution_char='=',
                        usekeys=None):
    """ Exports a dictionary of inputs to a string.

    Inputs
    ----------
    input_dict : dict
        Dictionary with the inputs to be exported to a string.
    entry_char : str
        Character used to start an input line in the config file.
    attrbution_char : str
        Character that separates the key name from its value.
    comment_char : str
        Character that indicates a commentary. Everything after this
        character is ignored.
    usekeys : list
        Use that input to select the input_dict entries that you want
        to export.
        Inform a list of the desired keys. Default is the whole dict.

    Returns
    -------
    result_str : str
        String with the formated inputs that were read from the input_dict.
    """
    # Selects the desired entries of the input_dict
    if usekeys is not None:
        input_dict = {key: input_dict[key] for key in usekeys}

    result_str = ""

    for key, value in input_dict.items():
        result_str += entry_string(key, value, entry_char, attribution_char)

    return result_str


def count_header_lines(file_name, header_end="-----\n"):
    """Opens and reads a file until it finds a 'header finalization' line.
    By standard, such line is '-----\n' (five -).
    Returns only the number of lines in the header.
    After that operation, the file is closed.

    Parameters
    ----------
    file_name : str
        Path for the (closed) file.
    header_end : str
        String that marks the end of a header section.
        Must contain the '\n' at the end.
    """
    fp = open(file_name, "r")

    for i, line in enumerate(fp):  # Reads file until eof
        # Checks if the line is a header end
        if line == header_end:
            fp.close()
            return i + 1

    # If EOF is reached without finding the header end, assumes there is no header
    print("Hey, warning: no 'header_end' line was found in file."
          "It will be assumed that the file has no header (i.e., 0 "
          "header lines).\n"
          "File path: '{}'\n"
          "The expected header end was: '{}'".format(file_name, header_end)
          )
    fp.close()
    return 0


def read_file_header(filename, header_end="-----\n"):
    """Reads a file until it finds a 'header finalization' line, and
    returns the read content.

    Parameters
    ----------
    filename : str
        Path for the file
    header_end : str
        String that marks the end of a header section.
        Must contain the '\n' at the end.

    Returns
    -------
    output : list
        A list with each line from the header. Does not contain the "header_end"
        line.
    """
    fp = open(filename, 'r')
    output = []

    # Reads file line once.
    line = fp.readline()

    while line:  # Reads until eof
        # Checks if the line is a header end
        if line == header_end:
            fp.close()
            return output
        output += [line[:-1]]  # Stores the line without the \n character
        line = fp.readline()

    # If EOF is reached without finding the header end, an error is raised.
    fp.close()
    raise EOFError("Hey, I did not find the header ending string on file:\n"
                   "File: '{}'\n"
                   "Ending str:'{}'\n".format(fp.name, header_end))
