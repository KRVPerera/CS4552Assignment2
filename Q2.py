import numpy as np
import scipy as sp
import sys
import random as rand
import logging
import math
from copy import deepcopy

epsilon = sys.float_info.epsilon
np.set_printoptions(formatter={'float': '{: 0.20f}'.format})

logLevel = logging.INFO

logging.basicConfig(level=logLevel, format=' %(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

def powerIterationMethod(matrix, startVector, iterations):
    result = startVector
    curIteration = 0
    actual = 11.0
    computedEigenValue = 0
    absoluteError = 100
    relativeError = 100
    tableWidth = 81
    print("-"*tableWidth)
    print("|  k  |      xkT \t\t\t\t      |  ||yk||infinity\t| computed eiegen value |")
    print("-"*tableWidth)
    while(abs(actual - computedEigenValue) > epsilon):
        result = matrix*result
        ykinf = np.linalg.norm(result, ord=np.inf)
        result = result/np.linalg.norm(result, ord=np.inf)
        # infinity normal is the maximum from absolute values of the vector elements

        curIteration += 1

        absoluteError = abs(computedEigenValue - actual);
        relativeError = absoluteError*100 / actual;

        computedEigenValue = (matrix*result).item(0)/result.item(0)


        print("| {0:2}  | [{1:6f} {2:6f} {3:6f}]  |  {4:}\t\t\t| {5} \t|".format(curIteration, result.item(0),result.item(1),result.item(2),
                                                                                 ykinf.item(0),computedEigenValue))
        if(curIteration > iterations):
            break

    print("-"*tableWidth)
    print("Absolute Error\t\t: {0}\nRelative Error (%)\t: {1} %".format(absoluteError, relativeError))
    return computedEigenValue


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
    logging.info("Random matrix generation done")
    return retMat


def main():
    # mat = np.matrix('2 3 2;10 3 4;3 6 1',dtype=float)
    # initial = np.matrix('0.0;0.0;1.0')
    # print("Matrix given in the assignment")
    # print(mat.tolist())
    #
    #
    # [eigens, vecs] = np.linalg.eig(mat)
    # print("Eiegen values of matrix A of assignment")
    # print(eigens)
    # print("Maximum actual eiegen value : {0}".format(round(max(eigens).item(0))))
    #
    # print("\nPower Iteration method")
    # print("Initial vector : {}".format(initial.tolist()))
    # computedValue = powerIterationMethod(mat, initial, 100)
    # print("Result from power iteration method : {}".format(computedValue))

    print("Creating matrix for section e")
    qemat = Q2PartESparseMatrix(1000)



if __name__ == '__main__':
    main()
