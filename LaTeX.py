#
# Utilities to help when working with LaTeX in Python
#

import numpy as np

def LaTeXMatrixToArray(matrix):
    """Prints the matrix in LaTeX form

    :matrix: numpy matrix
    :returns: String of the matrix

    """
    dims = matrix.shape
    # print(dims)  # dims[0]: height, dims[1]: width
    opening = r"\left[\begin{array}{" + "c" * dims[1] + r"}"
    closing = r"\end{array}\right]"
    string = ""
    for idx, x in enumerate(np.nditer(matrix)):
        string += str(float(x))
        if idx % dims[1] == dims[1] - 1:
            string += r"\\ "
        else:
            string += " & "
    return opening + string + closing

def LaTeXSingleLine(lst):
    """Puts strings from list into an nx1 tabular

    :lst: The list of strings to be put into the tabular
    :returns: stirng of tabular environment
    """
    start = r"\begin{tabular}{" + "c" * len(lst) + r"}"
    inner = " & ".join(lst)
    end = r"\end{tabular}"
    return start + inner + end

def LaTeXCenter(lst):
    """Prints contents of list in center environment

    Each element of the list is put on a separate line (should be LaTeX
    formatted strings)
    :returns: string with stuff in the center
    """
    start = r"\begin{center}" + '\n'
    inner = '    ' + r"\\ ".join(lst) + r"\\"
    end   = '\n' + r"\end{center}" + '\n'
    return start + inner + end
