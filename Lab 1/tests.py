import unittest
import numpy as np

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

    def test_permutation(self):
        number = 10
        # Permutation for numbers from 0 to 10
        permutation = np.random.permutation(number)
        print("test_permutation\n {} \n {}".format(number, permutation))

    def test_random_array_float(self):
        arr = np.random.rand(2)
        print("test_random_array_float\n {}".format(arr))

    def test_random_array_int(self):
        max = 10
        size = 2
        arr = np.random.randint(max, size=size)
        print("test_random_array_int\n {}".format(arr))

    def test_shuffle_array(self):
        size = 10
        arr = np.random.randint(10, size=size)
        permutation = np.random.permutation(size)
        # This is for ndarray.
        # Because there we have 2 dimensional array, and we shuffle only second one (columns for example)
        # shuffled_arr = arr[:, permutation]
        shuffled_arr = arr[permutation]
        print("test_shuffle_array\n {} \n {}".format(arr, shuffled_arr))

    def test_matmul(self):
        a = np.array([[1, 0], [0, 1]])
        b = np.array([[4, 1], [2, 2]])
        print("test_shuffle_array\n {} \n {}".format(np.matmul(a, b), np.matmul(b, a)))


if __name__ == '__main__':
    unittest.main()
