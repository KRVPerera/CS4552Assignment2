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

    def test_set_bugfromadd(self):
        mat = [[1, 1, 0], [1, 0, 0], [0, 1, 0]]
        csr1 = CSR(mat)
        print(mat)
        print(csr1)
        csr1.set(0, 2, 3)
        mat[0][2] = 3

        print(mat)
        print(csr1)
        for i in range(3):
            for j in range(3):
                print(i, j, mat[i][j], csr1.get(i, j))
                self.assertEquals(mat[i][j], csr1.get(i, j))

    def test_add(self):
        mat = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        n = len(mat)
        csr1 = CSR(mat)
        csr2 = CSR(mat)
        csr3 = csr1.add(csr1, csr2)

    def test_add_1(self):
        mat = [[0, 0, 1], [1, 0, 0], [0, 0, 0]]
        n = len(mat)
        csr1 = CSR(mat)
        csr2 = CSR(mat)
        csr3 = csr1.add(csr1, csr2)
        self.assertEquals(mat[0][0] * 2, csr3.get(0, 0))
        self.assertEquals(mat[0][1] * 2, csr3.get(0, 1))
        self.assertEquals(mat[0][2] * 2, csr3.get(0, 2))

    def test_add_2(self):
        mat = [[0, 0, 1], [1, 0, 0], [0, 0, 0]]
        mat2 = [[0, 0, 0], [0, 1, 0], [0, 1, 1]]
        n = len(mat)
        csr1 = CSR(mat)
        csr2 = CSR(mat2)
        csr3 = csr1.add(csr1, csr2)
        for i in range(3):
            for j in range(3):
                print(i, j, mat[i][j] + mat2[i][j], csr3.get(i, j))
                self.assertEquals(mat[i][j] + mat2[i][j], csr3.get(i, j))

    def test_add_3(self):
        mat = [[0, 0, 1], [1, 0, 0], [1, 1, 5]]
        mat2 = [[0, 1, 1], [0, 0, 0], [1, 1, 1]]
        n = len(mat)
        csr1 = CSR(mat)
        csr2 = CSR(mat2)
        csr3 = csr1.add(csr1, csr2)
        print("-----------mats")
        print(mat)
        print(mat2)
        print("-----------mats")
        for i in range(3):
            for j in range(3):
                print(i, j, mat[i][j] + mat2[i][j], csr3.get(i, j))
                self.assertEquals(mat[i][j] + mat2[i][j], csr3.get(i, j))

    def test_add_4(self):
        mat = [[1, 0, 0], [1, 0, 0], [0, 1, 0]]
        mat2 = [[0, 1, 1], [0, 0, 1], [1, 1, 1]]
        n = len(mat)
        csr1 = CSR(mat)
        csr2 = CSR(mat2)
        csr3 = csr1.add(csr1, csr2)
        print("-----------mats")
        print(mat)
        print(mat2)
        print("-----------mats")
        print("-----------Output")
        print(csr3)
        for i in range(3):
            for j in range(3):
                print(i, j, mat[i][j] + mat2[i][j], csr3.get(i, j))
                self.assertEquals(mat[i][j] + mat2[i][j], csr3.get(i, j))

    def test_getRow_0(self):
        mat = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        n = len(mat)
        csr1 = CSR(mat)
        for j in range(3):
            row0, ids = csr1.getRow(j)
            print(row0, ids, mat[j])
            for i, val in enumerate(mat[j]):
                if (val != 0):
                    self.assertEquals(val, row0[int(ids[i])])
