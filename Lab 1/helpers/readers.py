import mlxtend
from sklearn.datasets import fetch_openml
from mlxtend.data import mnist_data

class DatasetReader:
    def __init__(self, mnist_path="../dataset/mnist/"):
        self.MNIST_PATH = mnist_path
        self.train_images = self.MNIST_PATH + "train-images.idx3-ubyte"
        self.train_labels = self.MNIST_PATH + "train-labels.idx1-ubyte"
        self.test_images = self.MNIST_PATH + "t10k-images.idx3-ubyte"
        self.test_labels = self.MNIST_PATH + "t10k-labels.idx1-ubyte"

    def read_mnist_train_dataset(self):
        return mlxtend.data.loadlocal_mnist(self.train_images, self.train_labels)

    def read_mnist_test_dataset(self):
        return mlxtend.data.loadlocal_mnist(self.test_images, self.test_labels)


def get_mlxtend_mnist():
    X, y = mnist_data()
    X = X / 255
    return X, y


def get_sklearn_mnist():
    # from sklearn.datasets.base import get_data_home
    # print(get_data_home())
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version="1", return_X_y=True)
    X = X / 255
    return X, y
