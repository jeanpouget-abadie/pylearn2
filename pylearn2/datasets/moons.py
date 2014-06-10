from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy

class Moons(DenseDesignMatrix):
    def __init__(self, num_X, noise, n_neighbors=None):
        X, y = make_moons(n_samples=num_X, noise=noise)
        self.shape = (num_X, 2)
        if n_neighbors is not None:
            nbrs = NearestNeighbors(n_neighbors).fit(X)
            distances, indices = nbrs.kneighbors(X)
            super(Moons, self).__init__(X=(X,1,indices))
        else:
            super(Moons, self).__init__(X=X)

class One_Moon(DenseDesignMatrix):
	def __init__(self, num_X):
		numpy.random.seed(23)
		theta = numpy.random.uniform(low=0, high=2 * numpy.pi, size=(num_X, 1))
		X = numpy.concatenate((numpy.cos(theta), numpy.sin(theta)), axis=1)
		self.shape = (num_X, 2)
		super(One_Moon, self).__init__(X=X)