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
        np.array([2, 2], float)
        self.assertTrue(True)

class TestSigmoid(unittest.TestCase):

    def test_correct_calculating(self):
        x = 2.3
        

if __name__ == '__main__':
    unittest.main()