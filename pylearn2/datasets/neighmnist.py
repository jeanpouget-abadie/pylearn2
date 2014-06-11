from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.mnist import MNIST
from PIL import Image
import numpy

class NeighMNIST(DenseDesignMatrix):
    def __init__(self, n_neighbors):
        data = MNIST(which_set='train', start=0,
                     stop=50000, binarize='True')
        data = data.get_data()
        digits = data[0]
        labels = data[1]
        flatlabels = labels.flatten()
        index = numpy.argsort(flatlabels)
        sortedlabels = labels[index]
        sorteddigits = digits[index]
        X = [numpy.zeros((10, n_neighbors, 784)),
             numpy.zeros((10, n_neighbors, 1))]
        for i in xrange(10):
            k = 0
            while k < len(data[0]) and sortedlabels[k] != i:
                k = k+1
            print k
            for j in xrange(n_neighbors):
                X[0][i, j] = sorteddigits[k + j]
                X[1][i, j] = sortedlabels[k + j]
        self.shape = ((10, n_neighbors, 784), (10, n_neighbors, 1))
        super(NeighMNIST, self).__init__(X=X)
