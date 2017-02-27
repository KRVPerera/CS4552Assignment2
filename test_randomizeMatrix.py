from unittest import TestCase
from math import ceil
from math import floor
from Q1 import randomizeMatrix



class TestRandomizeMatrix(TestCase):

    def test_90PrecentZerosAll(self):
        for i in [10, 50, 100, 500, 1000, 5000]:
            with self.subTest(i==i):
                randMat = randomizeMatrix(i)
                nonZerosCount = 0
                total = 0
                for row in randMat:
                    nonZeros = [1 for i in row if i != 0.0]
                    nonZerosCount += sum(nonZeros)
                    total += len(row)
                self.assertEquals(90.0, floor((total - nonZerosCount) * 100.0 / total))