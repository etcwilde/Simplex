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

def printSingleLine(lst):
    """Puts strings from list into a tabular

    :lst: The list of strings to be put into the tabular
    :returns: stirng of tabular environment
    """
    start = r"\begin{tabular}{" + "c" * len(lst) + r"}"
    inner = " & ".join(lst)
    end = r"\end{tabular}"
    return start + inner + end


def basisVector(height, index):
    """Creates a 1xheight basis vector with the given index set to 1

    :height: Dimensions of the vector
    :index: Which dimension is the vector facing
    :returns: a basis vector in that direction
    """
    retval = np.zeros(height)
    retval[index] = 1
    return retval


class LinearSolver():

    """Solves linear programs"""

    def __init__(self, c, A, b):
        """
        Linear program solver

        :c: maximization function coefficients
        :A: Constraint matrix
        :b: Constraint coefficients

        """
        self._c = c
        self._A = A
        self._b = b

        self._solution = None

        self._N = [ i for i in range(self._A.shape[1]) ]
        self._B = [ i + len(self._N) for i in range(self._A.shape[0]) ]
        self._originalBasic = self._B[:]
        self._originalNonBasic = self._N[:]
        self._zn = -np.copy(self._c)
        self._xb = np.copy(self._b)

        self._slackMat = np.identity(len(self._B))
        self._A = np.append(self._A, self._slackMat, axis=1)


    def getBasicMatrix(self):
        """ Generate the basic matrix of the dictionary

        :returns: The basic matrix
        """
        return np.concatenate([self._A[:, i] for i in self._B], axis=1)

    def getNonBasicMatrix(self):
        """Generates the non-basic matrix

        :returns: The nonbasic matrix

        """
        return np.concatenate([self._A[:, i] for i in self._N], axis=1)

    @staticmethod
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

    @staticmethod
    def latexPrint(matrix):
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

    def printLaTeXIndecies(self):
        """prints the basic and nonbasic indices
        :returns: the string of LaTeX that is the basic and non-basic indices

        """
        return printSingleLine([
                r"$B$: ${" +
                r"\left\{\begin{array}{" + "c" * len(self._B) + r"}" +
                " & ".join([str(x + 1) for x in self._B]) +
                r"\end{array}\right\}" +
                r"}$",
                r"$N$: ${" +
                r"\left\{\begin{array}{" + "c" * len(self._N) + r"}" +
                " & ".join([str(x + 1) for x in self._N]) +
                r"\end{array}\right\}" +
                r"}$"
                ])

    def printLaTeXVaribles(self):
        return printSingleLine([
            r"$x_B$: $" + self.latexPrint(self._xb) + "$",
            r"$z_N$: $" + self.latexPrint(self._zn.transpose()) + "$"
            ])


    def printLaTeXBaseMatrices(self):
        """ prints the basic and nonbasic matrices
        :returns: the string of LaTeX that is the basic and non-basic matrices
        """
        return printSingleLine(["$B$: $" + self.latexPrint(self.getBasicMatrix()) + "$",
            "$N$: $" + self.latexPrint(self.getNonBasicMatrix()) + "$"])



    def __str__(self):
        return str(self._N) + str(self._B)

    def solve(self):
        """Solves the linear program

        :returns: Solution to linear program

        """

        iteration = 0
        while True:
            # Step 1 / 2
            print(r"\textbf{Iteration " + str(iteration + 1) + r": }" + "\n")

            print(r"\begin{center}")
            print(self.printLaTeXIndecies() + r"\\" + '\n')
            print(self.printLaTeXBaseMatrices() + r"\\" + '\n')
            print(self.printLaTeXVaribles() + r"\\" + '\n')
            print(r"\end{center}")


            print(r"\textit{Step 1:}")

            firstPivot = self.minCoeff(self._zn)
            if firstPivot[0] == -1:
                print("Coefficients of $z_n$ are all positive")
                return
                # print("Done optimizing", list(zip(self._N, np.nditer((self._zn)))))

            print("Negative Coefficient(s) in $z_n$")
            enteringIndex, value = firstPivot
            j = self._N[enteringIndex]

            print(r"\\\textit{Step 2:}\\")
            print(r"\begin{center}")
            print("$j$ = {}".format(j + 1))
            print(r"\end{center}")

            # Step 3

            elementary = np.matrix(basisVector(self.getNonBasicMatrix().shape[1],
                enteringIndex)).transpose()
            binv = np.linalg.inv(self.getBasicMatrix())
            deltaxb = binv * self.getNonBasicMatrix() * elementary

            print(r"\textit{Step 3:}\\")
            print(r"\begin{center}$\Delta x_B$ = $" + self.latexPrint(binv) +
                    self.latexPrint(self.getNonBasicMatrix()) +
                    self.latexPrint(elementary) + " = " +
                    self.latexPrint(deltaxb) + r"$\end{center}")


            # Step 4
            numerators = [num.item(0) for num in np.nditer(deltaxb)]
            denominators = [num.item(0) if num.item(0) is not 0 else None for num in np.nditer(self._xb)]

            nums = [numerators[idx] / val for idx, val in enumerate(denominators) if val is not None]
            vectorIndex, t = max(enumerate(nums), key=operator.itemgetter(1))
            t = 1 / t

            print(r"\textit{Step 4:}\\")
            print(r"\begin{center}")
            print("$t$ = {}".format(t))
            print(r"\end{center}")


            # Step 5
            leavingIndex = self._B[vectorIndex]

            print(r"\textit{Step 5:}\\")
            print(r"\begin{center}")
            print("$i$ = {}".format(leavingIndex + 1))
            print(r"\end{center}")


            # Step 6
            elementary = np.matrix(basisVector(self.getNonBasicMatrix().shape[0],
                vectorIndex)).transpose()
            binvNT = (binv * self.getNonBasicMatrix()).transpose()
            deltazn = -binvNT * elementary
            print(r"\textit{Step 6:}\\")
            print(r"\begin{center}$\Delta z_N = " + self.latexPrint(deltazn) +
                    r"$\end{center}")

            # Step 7
            s = self._zn.item(enteringIndex) / deltazn.item(enteringIndex)
            print(r"\textit{Step 7:}\\")
            print(r"\begin{center}$s = " + str(s) + r"$\end{center}")

            # Step 8

            self._xb = self._xb - t * deltaxb
            self._zn = self._zn.transpose() - s * deltazn


            originalXb = self.latexPrint(self._xb)
            originalzn = self.latexPrint(self._zn)

            leavingIndexIndex = self._B.index(leavingIndex)
            enteringIndexIndex = self._N.index(j)


            self._B[leavingIndexIndex] = j
            self._N[enteringIndexIndex] = leavingIndex



            self._xb.put(leavingIndexIndex, t)
            self._zn.put(enteringIndexIndex, s)
            self._zn = self._zn.transpose()

            print(r"\textit{Step 8:}\\")
            print(r"\begin{center}")

            print(
                    printSingleLine([
                        "$x_{}^* = {}$".format(leavingIndexIndex + 1, t),
                        "$x_B^* = " + originalXb + "$"
                        ]))
            print(r"\\")
            print(
                    printSingleLine([
                        "$z_{}^* = {}$".format(enteringIndexIndex + 1, s),
                        "$z_N^* = " + originalzn + "$"
                        ]))

            print(r"\end{center}")

            iteration += 1



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
    # print(pDic)
    lp = LinearSolver(np.matrix(pDic['c']), np.matrix(pDic['A']), np.matrix(pDic['b']))
    lp.solve()
    print()


if __name__ == "__main__":
    main()
