#!/bin/env python3

"""
The Simplex Algorithm

This is not an efficient version of the revised Simplex Algorithm. It is
designed with the intention of showing the steps involved in the simplex
method.
"""

import numpy as np
import argparse
import csv


class LinearProgram():

    """Linear program contains a constraint coefficent matrix, the constrain
    values, and the coefficients for the maximization problem"""

    def __init__(self, c, A, b):
        """Takes the inputs and converts them into a linear program

        Parameters
        ----------
        c: list of numbers
            Coefficients for the maximization function
        A: list of lists of numbers (float/int)
            Constraint coefficents
        b: list of numbers
            Constraint bounds
        """
        pass


def checkProgram(program_dictionary):
    """checkProgram
    ensures that the program is sane before we put it into the linear program
    class

    Parameters
    ----------
    program_dictionary: dictionary
        Dictionary containing the A, b, and c components of a linear program

    Returns
    ------
        Tuple,  First item is whether the program is sane
                Second item is the error message if there is one
    """
    if len(program_dictionary['A']) != len(program_dictionary['b']):
        return (False, "Constraint sizes don't match")
    for idx, r in enumerate(program_dictionary['A']):
        if len(r) != len(program_dictionary['c']):
            return (False,
                    "Row {0} in the constrain matrix is of the wrong size ({1})"
                    .format(idx, len(program_dictionary['c'])))
    return True,

def processInput(filenames):
    """processInput

    Takes the list of filenames and converts them into a linear program

    Parameters
    ----------
    filenames: list of strings
        List of filenames

    Returns
    -------
        Dictionary with the components for the linear program
    """
    csv_map = {0: 'A', 1: 'b', 2: 'c'}
    csv_contents = {'A': [], 'b': [], 'c': []}

    csv_actions = {
        0: (lambda i: [[float(item) for item in r] for r in i]),
        1: (lambda i: [float(item) for sublist in i for item in sublist if item != '']),
        2: (lambda i: [float(item) for sublist in i for item in sublist if item != '']),
        3: (lambda i: [float(item) for sublist in i for item in sublist if item != '']),
    }

    for f_idx, fname in enumerate(filenames):
        with open(fname, 'r') as f:
            try:
                dialect = csv.Sniffer().sniff(f.read(1023))
                f.seek(0)
                csv_reader = csv.reader(f, dialect)
            except:
                f.seek(0)
                csv_reader = csv.reader(f)
            csv_contents[csv_map[f_idx]] = \
                    csv_actions[f_idx]([row for row in csv_reader])
    return csv_contents


def main():
    """
    Runs the revised simplex method on the input files
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('files', metavar='F', type=str, nargs=3,
            help='List of files to process File 1: Constraint matrix, '+
            'File 2: b constraints, ' +
            'File 3: c coefficients')
    ap.add_argument('--version', action='version', version='%(prog)s 0.0')
    args = ap.parse_args()
    pDic = processInput(args.files)
    results = checkProgram(pDic)
    if not results[0]:
        print(results[1])
        return




if __name__ == "__main__":
    main()
