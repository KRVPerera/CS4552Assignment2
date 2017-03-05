from collections import defaultdict

import numpy as np
import random as rand
import math
import bisect
import logging
import argparse
from copy import deepcopy
import itertools
from scipy.sparse import csr_matrix

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
        row = int(row)
        col = int(col)
        self.set_new(row, col, value)


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
            id = bisect.bisect_left(self.col_ind, col, lo=start, hi=end)
            if (id == end):
                self.col_ind = np.insert(self.col_ind, id, col)
                self.val = np.insert(self.val, id, value)
                set = True
            elif (self.col_ind[id] == col):
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
        col_ind_tmp = []
        for row, j in enumerate(rows):
            for i in range(j):
                col_ind_tmp.append(row)

        ccs.col_ind = np.array(col_ind_tmp)
        # ccs.col_ind = np.concatenate((ccs.col_ind,col_ind_tmp))

        val_tmp = []
        for k, lis in dic.items():
            for id, value in enumerate(lis):
                val_tmp.append(value)

        ccs.val = np.concatenate((ccs.val,val_tmp))

        col_ptr_tmp = []
        sum = 0
        for i in range(self.W):
            sum += len(dic[i])
            col_ptr_tmp.append(sum)
        ccs.row_ptr = np.concatenate((ccs.row_ptr,col_ptr_tmp))
        str = "\nValues : {}\nIndices : {}\nIndices Pointers : {}\n".format(ccs.val, ccs.col_ind, ccs.row_ptr)
        logging.debug(str)
        return ccs

    def getRow(self, i):
        if (i + 1 > len(self.row_ptr)):
            raise IndexError
        start = self.row_ptr[i]
        end = self.row_ptr[i + 1]
        elms = end - start
        if (elms == 0):
            return [], []
        else:
            cols = []
            vals = []
            for i in range(elms):
                cols.append(self.col_ind[start + i])
                vals.append(self.val[start + i])
            assert len(cols) == len(vals)
            return vals, cols

    def add(self, csr1, csr2):
        ccs = deepcopy(csr1)
        rows1 = csr1.row_ptr
        maxL = len(rows1)
        # if(maxL != len(rows2)): # matrix lengths are different
        #     raise IndexError("Cannot add matrices with different dimentions")

        for i in range(maxL - 1):
            row1_c, row1_col_ids = csr1.getRow(i)
            row2_c, row2_col_ids = csr2.getRow(i)
            row2_len = len(row2_col_ids)
            row1_len = len(row1_col_ids)
            if row2_len == 0:
                continue
            elif row1_len == 0:
                for col, val in zip(row2_col_ids, row2_c):
                    ccs.set(i, col, val)
            elif row2_len > 0:
                k = 0
                for col, val in zip(row2_col_ids, row2_c):
                    if k < row1_len:
                        col1_id = row1_col_ids[k]
                        if (col1_id == col):
                            ccs.set(i, col, val + row1_c[k])
                            k = k + 1
                        elif col1_id < col:
                            oldk = k
                            k = k + 1
                            ccs.set(i, col, val)
                        else:
                            ccs.set(i, col, val)
                    else:
                        ccs.set(i, col, val)
        return ccs

    def vecmul(self, vec):
        side = len(self.row_ptr) - 1
        vside = len(vec)
        if (side != vside):
            raise IndexError("Cannot multiple these dimensions")
        ans = np.zeros((side, 1))
        for i in range(side):
            ithRow_vals, ithRow_ids = self.getRow(i)
            for col, val in zip(ithRow_ids, ithRow_vals):
                ans[i][0] += val * vec[int(col)][0]
        return ans



    def __str__(self, *args, **kwargs):
        str = "Values : {}\nIndices : {}\nIndices Pointers : {}\n".format(self.val, self.col_ind, self.row_ptr)
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

    print("Randomly generated matrix : ")
    print(mat4)

    csr = CSR(mat4)
    print("\nCSR Format of the matrix : ")
    print(csr)

    logging.debug("Changing diagonal elements to 2016")
    for i in range(N):
        mat4[i][i] = 2016.0
        csr.set(i, i, 2016.0)

    print("After Changing the diagonal elements to 2016")
    print(mat4)

    print("\nCSR Format of the matrix : ")
    print(csr)

    print("Changing the CSR format to CCS")
    csc_me = csr.toCCS()
    print(csc_me)


def GetMatVectMultTimings():
    print("Testing mat-vec multiplication")
    initial = np.ones((3, 1))
    # initial[1][0] = 5
    matt = randomizeMatrix(3)
    csrt = CSR(matt)
    ans = csrt.vecmul(initial)
    print(ans)
    raise NotImplemented


def GetMatMatAddTimings():
    raise NotImplemented

if __name__ == '__main__':
    main()