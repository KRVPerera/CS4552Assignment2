import numpy as np
import scipy as sp
import sys
import random as rand
import logging
import math
from copy import deepcopy
from scipy.sparse import csr_matrix

epsilon = sys.float_info.epsilon
np.set_printoptions(formatter={'float': '{: 0.20f}'.format})

logLevel = logging.INFO
logging.basicConfig(level=logLevel, format=' %(asctime)s - %(levelname)s - %(funcName)s - %(message)s')


def powerIterationMethod(matrix, startVector, iterations, sparse=True):
    if (sparse):
        return __powerIterationMethodSparse(matrix, startVector, iterations)
    else:
        return __powerIterationMethodDense(matrix, startVector, iterations)


def __powerIterationMethodDense(matrix, startVector, iterations):
    result = startVector
    curIteration = 0

    computedEigenValue = 0
    computedEigenValueOld = 1
    tableWidth = 81
    print("-"*tableWidth)
    print("|  k  |      xkT \t\t\t\t      |  ||yk||infinity\t| computed eiegen value |")
    print("-"*tableWidth)
    while (abs(computedEigenValueOld - computedEigenValue) > epsilon):
        result = matrix*result
        ykinf = np.linalg.norm(result, ord=np.inf)
        result = result/np.linalg.norm(result, ord=np.inf)
        # infinity normal is the maximum from absolute values of the vector elements
        computedEigenValueOld = computedEigenValue
        curIteration += 1

        try:
            computedEigenValue = (matrix * result).item(0) / result.item(0)
        except ZeroDivisionError:
            pass


        print("| {0:2}  | [{1:6f} {2:6f} {3:6f}]  |  {4:}\t\t\t| {5} \t|".format(curIteration, result.item(0),result.item(1),result.item(2),
                                                                                 ykinf.item(0),computedEigenValue))
        if(curIteration > iterations):
            break

    print("-"*tableWidth)
    return matrix * result


def __powerIterationMethodSparse(matrix, startVector, iterations):
    result = startVector
    curIteration = 0
    matrix = csr_matrix(matrix)
    computedEigenValue = 0
    computedEigenValueOld = 1
    tableWidth = 81
    print("-" * tableWidth)
    print("|  k  |      xkT \t\t\t\t      |  ||yk||infinity\t| computed eiegen value |")
    print("-" * tableWidth)
    while (abs(computedEigenValueOld - computedEigenValue) > epsilon):
        result = matrix * result
        ykinf = np.linalg.norm(result, ord=np.inf)
        result = result / np.linalg.norm(result, ord=np.inf)
        # infinity normal is the maximum from absolute values of the vector elements
        computedEigenValueOld = computedEigenValue
        curIteration += 1

        try:
            computedEigenValue = (matrix * result).item(0) / result.item(0)
        except ZeroDivisionError:
            logging.error("Division by zerp")

        print("| {0:2}  | [{1:6f} {2:6f} {3:6f}]  |  {4:}\t\t\t| {5} \t|".format(curIteration, result.item(0),
                                                                                 result.item(1), result.item(2),
                                                                                 ykinf.item(0), computedEigenValue))
        if (curIteration > iterations):
            break

    print("-" * tableWidth)
    return matrix * result

def Q2PartESparseMatrix(n):
    retMat = np.zeros(dtype=float, shape=[n, n])
    randCounts = np.empty(shape=n, dtype=int)

    for i in range(n):
        randCounts[i] = (rand.random() * 100) % 16 + 5

    randCounts_copy = deepcopy(randCounts)
    logging.debug(randCounts)
    while (sum(randCounts) > 0):
        col = math.floor((rand.random() * 1000) % n)
        if (randCounts[col] > 0):
            row = math.floor((rand.random() * 1000) % n)
            randCounts[col] = randCounts[col] - 1
            retMat[row][col] = 1.0 / randCounts_copy[col]

    logging.debug(retMat)
    logging.debug("Random matrix generation done")
    return retMat


def pageRank(mat):
    n = len(mat)
    randVec = np.empty(shape=n, dtype=float)
    for i in range(n):
        randVec[i] = (100.0 / n)

    pages = powerIterationMethod(mat, randVec, 10)
    return pages


def main():
    rand.seed(20)
    mat = np.matrix('2 3 2;10 3 4;3 6 1', dtype=float)
    initial = np.matrix('0.0;0.0;1.0')
    print("Matrix given in the assignment")
    print(mat.tolist())

    [eigens, vecs] = np.linalg.eig(mat)
    print("Eiegen values of matrix A of assignment")
    print(eigens)
    print("Maximum actual eiegen value : {0}".format(round(max(eigens).item(0))))

    print("\nPower Iteration method")
    print("Initial vector : {}".format(initial.tolist()))
    computedValue = powerIterationMethod(mat, initial, 100)
    print("Result from power iteration method : {}".format(computedValue))

    print("Creating matrix for section e")
    qemat = Q2PartESparseMatrix(1000)
    print("Matrix creation ok...Calculating page ranks")
    pageRanks = pageRank(qemat)
    logging.debug(pageRanks)
    for i, rank in enumerate(pageRanks):
        print("Page {0:4}  :   Rank - {1} ".format(i + 1, rank))


if __name__ == '__main__':
    main()
