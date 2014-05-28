from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils.iteration import resolve_iterator_class
import numpy

class BernoulliPrior(DenseDesignMatrix):
    def __init__(self, num_z, dim_size):
        self.shape = (num_z, dim_size)
        assert num_z <= 500  # Memory overflow otherwise (dim_size * num_z * n_vis * 4 bytes)
        X = numpy.random.randint(2, size=(num_z, dim_size)).astype('float32')
        # self._iter_subset_class = resolve_iterator_class('shuffled_sequential')
        super(BernoulliPrior, self).__init__(X=X)
