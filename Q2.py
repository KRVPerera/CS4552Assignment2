import numpy as np
import scipy as sp
import sys

epsilon = sys.float_info.epsilon
np.set_printoptions(formatter={'float': '{: 0.20f}'.format})

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



def main():
    mat = np.matrix('2 3 2;10 3 4;3 6 1',dtype=float)
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

if __name__ == '__main__':
    main()
