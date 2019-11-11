import matplotlib
import matplotlib.pyplot as plt


def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title('true label: %d' % y[idx])
    plt.show()


def plot_digit2(image28x28, label=""):
    reshapedIm = image28x28.reshape(28, 28)
    plt.imshow(reshapedIm, cmap=matplotlib.cm.binary)
    plt.title('Label: %s' % label)
    plt.axis("off")
    plt.show()
