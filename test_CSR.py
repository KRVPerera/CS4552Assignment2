from unittest import TestCase
from unittest import skip
import random
from scipy import sparse
from Q1 import CSR, randomizeMatrix


class TestCSR(TestCase):

    def test_load(self):
        for n in [10, 50, 100, 500]:
            with self.subTest(i=n):
                mat2 = randomizeMatrix(n)
                csr_sci = sparse.csr_matrix(mat2)
                csr = CSR()
                csr.load(mat2)

                for index, i in enumerate(csr_sci.data):
                    self.assertEquals(i, csr.val[index])
                    self.assertEquals(csr.col_ind[index], csr_sci.indices[index])

                for id,j in enumerate(csr_sci.indptr):
                    self.assertEquals(j, csr.row_ptr[id])

    def test_get(self):
        for n in [10, 50, 100, 500]:
            with self.subTest(i=n):
                mat2 = randomizeMatrix(n)
                csr_sci = sparse.csr_matrix(mat2)
                csr = CSR()
                csr.load(mat2)

                for i in range(n):
                    for j in range(n):
                        self.assertEquals(mat2[i][j], csr.get(i,j))

    def test_get_simple(self):
        mat = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
        n = 4
        csr = CSR(mat)
        for i in range(n):
            for j in range(n):
                print(i,j)
                self.assertEquals(mat[i][j], csr.get(i, j))



    def test_set(self):
        for n in [4, 10, 50, 100]:
            mat2 = randomizeMatrix(n)
            csr = CSR(mat2)

            for i in range(n):
                for j in range(n):
                    rand = random.random()
                    mat2[i][j] = rand
                    csr.set(i, j, rand)
                    # self.assertEquals(rand, csr.get(i,j))

            csr2 = CSR(mat2)
            for i in range(n):
                for j in range(n):
                    self.assertEquals(mat2[i][j], csr2.get(i, j))

    def test_set_Diag(self):
        for n in [4, 10, 50, 100]:
            mat2 = randomizeMatrix(n)
            csr = CSR()
            csr.load(mat2)
            for i in range(n):
                rand = random.random()
                mat2[i][i] = rand
                csr.set(i, i, rand)
                print(mat2)
                self.assertEquals(rand, csr.get(i, i))

            csr2 = CSR(mat2)
            for i in range(n):
                for j in range(n):
                    self.assertEquals(mat2[i][j], csr2.get(i, j))


    def test_set_bug(self):
        mat = [[1, 0, 0, 0], [1, 0, 0, 0.5], [1, 0, 0, 0], [1, 0, 0, 0]]
        n = 4
        csr = CSR(mat)
        csr_sci = sparse.csr_matrix(mat)

        rand = random.random()
        mat[1][1] = rand
        csr.set(1, 1, rand)
        csr_sci[1,1] = rand

        for index, i in enumerate(csr_sci.data):
            self.assertEquals(i, csr.val[index])
            self.assertEquals(csr.col_ind[index], csr_sci.indices[index])

        for id, j in enumerate(csr_sci.indptr):
            self.assertEquals(j, csr.row_ptr[id])

    def test_set_bug_overwrite(self):
        mat = [[1, 0, 0, 0], [1, 0, 0, 0.5], [1, 0, 0, 0], [1, 0, 0, 0]]
        n = 4
        csr = CSR(mat)
        csr_sci = sparse.csr_matrix(mat)

        rand = random.random()
        mat[1][1] = rand
        csr.set(1, 1, rand)
        csr_sci[1,1] = rand

        print("------------ After first insertion")
        print(csr)
        print(csr_sci.data)
        print(csr_sci.indices)
        print(csr_sci.indptr)

        rand = random.random()
        mat[1][1] = rand
        csr.set(1, 1, rand)
        csr_sci[1, 1] = rand

        print("------------ After second insertion overwrite")
        print(csr)
        print(csr_sci.data)
        print(csr_sci.indices)
        print(csr_sci.indptr)

        for index, i in enumerate(csr_sci.data):
            self.assertEquals(i, csr.val[index])
            self.assertEquals(csr.col_ind[index], csr_sci.indices[index])

        for id, j in enumerate(csr_sci.indptr):
            self.assertEquals(j, csr.row_ptr[id])


    def test_set_bug_2(self):
        mat = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0]]
        n = 4
        csr = CSR()
        csr.load(mat)

        for i in range(n):
            for j in range(n):
                self.assertEquals(mat[i][j], csr.get(i, j))
        rand = random.random()

        print(mat)
        print(csr)

        mat[1][1] = rand
        csr.set(1, 1, rand)

        print(mat)
        print(csr)
        self.assertEquals(mat[1][0], csr.get(1, 0))

        print(mat)
        print(csr)

        for i in range(n):
            for j in range(n):
                self.assertEquals(mat[i][j], csr.get(i, j))

    def test_set_bug_diag(self):
        mat = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        n = len(mat)
        csr = CSR(mat)
        print(mat)
        print(csr)

        i, j = 1, 1
        rand = random.random()
        mat[i][j] = rand
        csr.set(i, j, rand)
        print(mat)
        print(csr)
        self.assertEquals(mat[i][j], csr.get(i, j))
