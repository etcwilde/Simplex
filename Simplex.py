#!/bin/env python3

"""
The Simplex Algorithm

This is not an efficient version of the revised Simplex Algorithm. It is
designed with the intention of showing the steps involved in the simplex
method.
"""

import argparse
import csv
import numpy as np
import numpy.linalg
import operator
import LaTeX

def checkProgram(program_dictionary):
    """checkProgram
    ensures that the program is sane before we put it into the linear program
    class

    :program_dictionary: dictionary
        Dictionary containing the A, b, and c components of a linear program

    :returns: Tuple,  First item is whether the program is sane
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

    :filenames: list of strings List of filenames
    :returns: Dictionary with the components for the linear program
    """
    csv_map = {0: 'A', 1: 'b', 2: 'c'}
    csv_contents = {'A': [], 'b': [], 'c': []}

    csv_actions = {
        0: (lambda i: [[float(item) for item in r] for r in i]),
        1: (lambda i: [[float(item)] for sublist in i for item in sublist if item != '']),
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

def basisVector(height, index):
    """Creates a 1xheight basis vector with the given index set to 1

    :height: Dimensions of the vector
    :index: Which dimension is the vector facing
    :returns: a basis vector in that direction
    """
    retval = np.zeros(height)
    retval[index] = 1
    return retval

def minCoeff(zCoeff):
    """Grabs most negative coefficient

    -- If index is -1, then we are done optimizing

    :zCoeff: The coefficients for the objective function
    :returns: index and most negative coefficient
    """
    retVal = (-1, 0)
    for idx, i in enumerate(np.nditer(zCoeff)):
        if i < retVal[1]:
            retVal = (idx, i)
    return retVal

def solveLinearProgram(c, A, b, debug=False):
    """solveLinearProgram

    Solve (maximizes) the linear program
    :c: Coefficients for the optimization function
    :A: Constraint Coefficient matrix
    :b: Constraint boundaries
    :returns: xb, zn, assignment, and solution to the linear program

    """
    if type(c) is not np.matrix:
        c = np.matrix(c)

    if type(A) is not np.matrix:
        A = np.matrix(A)

    if type(b) is not np.matrix:
        b = np.matrix(b)

    N = [i for i in range(A.shape[1])]
    B = [i + len(N) for i in range(A.shape[0])]

    zn = -c
    xb = b

    originalBasic = B[:]
    originalNonBasic = N[:]
    A = np.append(A, np.identity(len(B)), 1)

    def getBasicMatrix():
        """ Generate the basic matrix of the dictionary

        :returns: The basic matrix
        """
        return np.concatenate([A[:, i] for i in B], 1)

    def getNonBasicMatrix():
        """Generates the non-basic matrix

        :returns: The nonbasic matrix

        """
        return np.concatenate([A[:, i] for i in N], 1)

    if debug:
        iterations = 1
    while True:

        if debug:
            # Print basic info
            print(r"\textbf{Iteration " + str(iterations) + r": }" + "\n")
            BIndex = r"$\mathcal{B} = " + \
                    r"\left\{\begin{array}{" + "c" * len(B) + r"}" +\
                    " & ".join([str(x + 1) for x in B]) + \
                    r"\end{array}\right\}$"
            NIndex = r"$\mathcal{N} = " + \
                    r"\left\{\begin{array}{" + "c" * len(N) + r"}" +\
                    " & ".join([str(x + 1) for x in N]) + \
                    r"\end{array}\right\}$"
            print(LaTeX.LaTeXCenter([
                LaTeX.LaTeXSingleLine([BIndex, NIndex]), # The indices
                LaTeX.LaTeXSingleLine([
                    "$N = " + LaTeX.LaTeXMatrixToArray(getBasicMatrix()) + "$",
                    "$B = " + LaTeX.LaTeXMatrixToArray(getNonBasicMatrix()) + "$"
                    ]),  # The matrices
                LaTeX.LaTeXSingleLine([ "$x_B^* = " + LaTeX.LaTeXMatrixToArray(xb) + "$",
                    "$z_n^* = " + LaTeX.LaTeXMatrixToArray(zn.transpose()) + "$"
                    ]) # xb* and zn* vectors
                ]))

        if debug:
            print(r"\textit{Step 1:}")

        # Step 1/2
        firstPivot = minCoeff(zn)
        if firstPivot[0] == -1:
            var_assign = {idx : xb[B.index(n)].tolist()[0][0] if n in B else zn.transpose()[N.index(n)].tolist()[0][0] for idx, n in enumerate(originalNonBasic)}
            slack = {idx + len(originalNonBasic): zn.transpose()[N.index(n)].tolist()[0][0] if n in N else xb[B.index(n)].tolist()[0][0]  for idx, n in enumerate(originalBasic)}
            solution = sum([var_assign[index] * x for index, x in enumerate(np.nditer(c))])
            if debug:
                print("$z_n$ does not contain any negative values. Solution: {}".format(solution))

            return {'xb': list(zip(B, [x for r in xb.tolist() for x in r])),
                    'zn': list(zip(N, [x for r in zn.tolist() for x in r])),
                    'originalVars': originalNonBasic,
                    'assignment': var_assign,
                    'slack': slack,
                    'solution': solution
                    }


        enteringIndex, value = firstPivot
        j = N[enteringIndex]

        if debug:
            print("$z_n$ contains negative values. Continue optimizing.\n")
            print(r"\textit{Step 2:}\\")
            print(LaTeX.LaTeXCenter(['$j = {}$'.format(j + 1)]))

        # Step 3
        elementary = np.matrix(basisVector(getNonBasicMatrix().shape[1],
            enteringIndex)).transpose()
        binv = np.linalg.inv(getBasicMatrix())
        deltaxb = binv * getNonBasicMatrix() * elementary

        if debug:
            print(r"\textit{Step 3:}\\")
            print("$$\Delta x_b = {} {} = {}$$".format(
                LaTeX.LaTeXMatrixToArray(binv),
                LaTeX.LaTeXMatrixToArray(elementary),
                LaTeX.LaTeXMatrixToArray(deltaxb)))

        # Step 4
        numerators = [num.item(0) for num in np.nditer(deltaxb)]
        denominators = [num.item(0) if num.item(0) is not 0 else None for num in np.nditer(xb)]

        nums = [numerators[idx] / val for idx, val in enumerate(denominators) if val is not None]
        vectorIndex, t = max(enumerate(nums), key=operator.itemgetter(1))
        t = 1 / t

        if debug:
            print(r"\textit{Step 4:}\\")
            print("$$t = {}$$".format(t))

        # Step 5
        leavingIndex = B[vectorIndex]

        if debug:
            print(r"\textit{Step 5:}\\")
            print("$$i = {}$$".format(leavingIndex + 1))

        # Step 6
        elementary = np.matrix(basisVector(getNonBasicMatrix().shape[0],
            vectorIndex)).transpose()
        binvNT = (binv * getNonBasicMatrix()).transpose()
        deltazn = -binvNT * elementary

        if debug:
            print(r"\textit{Step 6:}\\")
            print("$$\Delta z_n = {} {} = {}$$".format(
                LaTeX.LaTeXMatrixToArray(binvNT),
                LaTeX.LaTeXMatrixToArray(elementary),
                LaTeX.LaTeXMatrixToArray(deltazn)))

        # Step 7
        s = zn.item(enteringIndex) / deltazn.item(enteringIndex)

        if debug:
            print(r"\textit{Step 7:}\\")
            print("$$s = {}$$".format(s))

        # Step 8
        xb = xb - t * deltaxb
        zn = zn.transpose() - s * deltazn

        if debug:
            oldXB = LaTeX.LaTeXMatrixToArray(xb)
            oldZN = LaTeX.LaTeXMatrixToArray(zn)

        leavingIndexIndex = B.index(leavingIndex)
        enteringIndexIndex = N.index(j)

        B[leavingIndexIndex] = j
        N[enteringIndexIndex] = leavingIndex

        xb.put(leavingIndexIndex, t)
        zn.put(enteringIndexIndex, s)

        if debug:
            print(r"\textit{Step 8:}\\")

            print(LaTeX.LaTeXCenter([
                LaTeX.LaTeXSingleLine([
                    "$x_{}^* = {}$".format(leavingIndex + 1, t),
                    "$x_B^* = " + oldXB + "$"
                    ]),
                LaTeX.LaTeXSingleLine([
                    "$z_{}^* = {}$".format(enteringIndexIndex + 1, s),
                    "$z_N^* = " + oldZN + "$"
                    ])
                ]))
            iterations += 1

        zn = zn.transpose()

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
    ap.add_argument('-d', '--debug', action='store_true')
    args = ap.parse_args()
    pDic = processInput(args.files)
    results = checkProgram(pDic)
    if not results[0]:
        print(results[1])
        return

    sol = solveLinearProgram(pDic['c'], pDic['A'], pDic['b'], debug=args.debug)
    if not args.debug:
        print(sol)


if __name__ == "__main__":
    main()
