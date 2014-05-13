from sklearn.datasets import make_moons

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


class Moons(DenseDesignMatrix):
    def __init__(self, num_X, noise):
        X, y = make_moons(n_samples=num_X, noise=noise)
        super(Moons, self).__init__(X=X)
