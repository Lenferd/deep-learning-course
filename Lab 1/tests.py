import unittest
import numpy as np


# class TestStringMethods(unittest.TestCase):
#
#     def test_upper(self):
#         self.assertEqual('foo'.upper(), 'FOO')
#
#     def test_isupper(self):
#         self.assertTrue('FOO'.isupper())
#         self.assertFalse('Foo'.isupper())
#
#     def test_split(self):
#         s = 'hello world'
#         self.assertEqual(s.split(), ['hello', 'world'])
#         # check that s.split fails when the separator is not a string
#         with self.assertRaises(TypeError):
#             s.split(2)

class TestNumpy(unittest.TestCase):
    def test_generate_array(self):
        arr = np.array([2, 2], float)
        print("test_generate_array\n {}".format(arr))

    def test_matrix_not_rec_creation(self):
        matrix = np.matrix([[1, 2], [3, 4]])
        print("test_matrix_not_rec_creation\n {}".format(matrix))

    def test_matrix_rec_creation(self):
        matrix = np.array([[1, 2], [3, 4]])
        print("test_matrix_rec_creation\n {}".format(matrix))

    def test_transpose(self):
        matrix = np.array([[1, 2], [3, 4]])
        print("test_transpose\n {} \n {}".format(matrix, matrix.T))

class TestSigmoid(unittest.TestCase):

    def test_correct_calculating(self):
        x = 2.3
        

if __name__ == '__main__':
    unittest.main()
