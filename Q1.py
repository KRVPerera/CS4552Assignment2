from collections import defaultdict

import numpy as np
import random as rand
import math
import bisect
import logging
import argparse
from scipy import sparse

logLevel = logging.ERROR

logging.basicConfig(level=logLevel, format=' %(asctime)s - %(levelname)s - %(funcName)s - %(message)s')


class CSR:
    """
        Complex Row Storage, Complex Sparse Row, Yale Format implementation for 2D sparse matrices
    """

    def __init__(self, denseMatrix=[[]]):
        self.load(denseMatrix)

    def load(self, denseMatrix):
        """
            Get a 2D dense matrix format and load the class with CSR format
        :param denseMatrix: 2D dense matrix
        """
        nonzeroCount = np.count_nonzero(denseMatrix)
        self.val = np.zeros(dtype=float, shape=[nonzeroCount])
        self.col_ind = np.zeros(dtype=float, shape=[nonzeroCount])
        self.row_ptr = [0]

        self.H = len(denseMatrix)
        self.W = len(denseMatrix[0])

        i = 0
        for row in denseMatrix:
            for colId, value in enumerate(row):
                if (value != 0):
                    # self.val.append(value)
                    self.val[i] = value
                    self.col_ind[i] = colId
                    i = i + 1
            self.row_ptr.append(i)
        logging.debug(self)

    def get(self, row, col):
        """
            Get the value at row,col location in the matrix. Row, col values are assumed to be indexes of 2D dense
            representation of the same matrix
        :param row: row index of the dense matrix form (i)
        :param col: column index of the dense matrix form (j)
        :return: value at the row, col (i,j) location a(i, j)
        """
        start, end, stride = self.row_ptr[row], self.row_ptr[row + 1], self.row_ptr[row + 1]-self.row_ptr[row ]

        if stride == 0:
            return 0
        elif stride == 1 and self.col_ind[start] == col:
            return self.val[start]
        else:
            """  Old code sequential search
            # id = -1
            # for j in range(start, end):
            #     if(self.col_ind[j] == col):
            #         id = j
            #         break
            # else:
            #     return 0
            # return self.val[id] """

            # New code uses binary search
            id = bisect.bisect_left(self.col_ind, col, lo=start, hi=end-1)
            if (self.col_ind[id] == col):
                return self.val[id]
        return 0

    def set(self, row, col, value):
        self.set_new(row, col, value)
        #start, end, stride = self.row_ptr[row], self.row_ptr[row + 1], self.row_ptr[row + 1] - self.row_ptr[row]
        # start, end, stride = self.__get__slice(row)
        # if stride != 0:
        #     id = bisect.bisect_left(self.col_ind, col, lo=start, hi=min(end , len(self.col_ind) - 1))
        #     if ((self.col_ind[id+1] == col) and ((id+1) <= len(self.col_ind) - 1)):
        #         self.val[id+1] = value
        #     else:
        #         self.col_ind.insert(id, col)
        #         self.val.insert(id, value)
        # elif stride == 1:
        #     pass
        # else:
        #     self.col_ind.insert(start, col)
        #     self.val.insert(start, value)
        # self.row_ptr[row + 1:] = map(lambda x: x + 1, self.row_ptr[row + 1:])

    def set_new(self, row, col, value):
        start, end, stride = self.__get__slice(row)
        set = False
        if stride == 0:
            self.col_ind = np.insert(self.col_ind, start, col)
            self.val = np.insert(self.val, start, value)
            set = True
        elif stride == 1:
            if (self.col_ind[start] == col):
                self.val[start] = value
            elif(self.col_ind[start] > col):
                self.col_ind = np.insert(self.col_ind, start, col)
                self.val = np.insert(self.val, start, value)
                set = True
            else:
                self.col_ind = np.insert(self.col_ind, start+1, col)
                self.val = np.insert(self.val, start+1, value)
                set = True
        else:
            id = bisect.bisect_left(self.col_ind, col, lo=start, hi=end - 1)
            if (self.col_ind[id] == col):
                self.val[id] = value
            else:
                self.col_ind = np.insert(self.col_ind, id, col)
                self.val = np.insert(self.val, id, value)
                set = True
        if(set):
            self.row_ptr[row + 1:] = map(lambda x: x + 1, self.row_ptr[row + 1:])

    def toCCS(self):
        ccs = CSR()
        ccs.row_ptr = [0]
        dic = defaultdict(list)
        for id, i in enumerate(self.col_ind):
            dic[i].append(self.val[id])

        rows = []
        for i in range(1, len(self.row_ptr)):
            rows.append(self.row_ptr[i] - self.row_ptr[i - 1])

        logging.debug(rows)
        for row, j in enumerate(rows):
            for i in range(j):
                ccs.col_ind.append(row)

        for k, lis in dic.items():
            for id, value in enumerate(lis):
                ccs.val.append(value)

        sum = 0
        for i in range(self.W):
            sum += len(dic[i])
            ccs.row_ptr.append(sum)
        str = "\nValues : {}\nrow_ind : {}\ncol_ptr : {}\n".format(ccs.val, ccs.col_ind, ccs.row_ptr)
        logging.debug(str)
        return ccs


    def __str__(self, *args, **kwargs):
        str = "Values : {}\ncol_ind : {}\nrow_ptr : {}\n".format(self.val, self.col_ind, self.row_ptr)
        return str

    def __get__slice(self,row):
        return (self.row_ptr[row], self.row_ptr[row + 1], self.row_ptr[row + 1] - self.row_ptr[row])

def randomizeMatrix(n):
    """
        This method will take n as a input and create a matrix with 10% non zero floats
        of size nxn and return the matrix to the calling method
        :param n: dimension of nxn matrix
    """
    retMat = np.zeros(dtype=float, shape=[n, n])
    height = len(retMat)
    width = len(retMat[0])
    totalSize = height * width
    nonZeroFraction = math.ceil(totalSize * 10.0 / 100)
    logging.info("Total number of elements \t\t: {0}".format(totalSize))
    logging.info("Nonzero fraction of elements \t: {0}".format(nonZeroFraction))
    while (nonZeroFraction > 0):
        i = math.floor(rand.random() * height) % height
        j = math.floor(rand.random() * width) % width
        retMat[i][j] = rand.random() * 10
        nonZeroFraction = nonZeroFraction - 1
    logging.debug("Final matrix {0}".format(retMat))
    logging.info("Random matrix generation done")
    return retMat


def main():
    parser = argparse.ArgumentParser(description="CS4552 Scientific Computing\nAssignment 2\nQ1", epilog="Thank you...")
    parser.add_argument("--n", "-n", type=int, help="Dimensions of the matrix")
    args = parser.parse_args()
    N = args.n
    logging.debug("Running Q1 with size : {0} x {0} matrix".format(N))

    mat4 = randomizeMatrix(N)
    # print("Randomly generated matrix : ")
    # print(mat4)
    # for i in range(N):
    #     logging.debug(mat4[i])
    #
    # csr = CSR(mat4)
    # print("\nCSR Format of the matrix : ")
    # print(csr)
    #
    # logging.debug("Changing diagonal elements to 2016")
    # for i in range(N):
    #     mat4[i][i] = 2016
    #     csr.set(i, i, 2016)
    #     print(mat4)
    #     print(csr)
    #
    # print("\nDiagonal set CSR Format of the matrix : ")
    # print(mat4)
    # print(csr)
    #
    # for i in range(N):
    #     for j in range(N):
    #         print("{0:-10}".format(csr.get(i, j)), end='\t')
    #     print()

    mtx = sparse.csr_matrix(mat4)
    print(mtx.data)
    print(mtx.indices)
    print(mtx.indptr)


if __name__ == '__main__':
    main()