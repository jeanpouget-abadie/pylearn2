import cPickle
from pylearn2.datasets.mnist import MNIST
from pylearn2.utils.rng import make_theano_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import function, tensor, scan
from pylearn2.utils import image
import itertools
import numpy

def sample_from_chain(model, x, rng):
    return model.gibbs_step_for_v(x, rng)[0]

def import_model():
    with open('binary_RBM.pkl', 'r') as f:
        model = cPickle.load(f)
    mnist = MNIST('train')
    rng = RandomStreams(123)
    return model, rng, mnist

def visualise_samples(n_samples, n_steps):
    model, rng, mnist = import_model()
    sample = tensor.fmatrix()
    output, updates = scan(fn=lambda i, x_init: model.gibbs_step_for_v(x_init, rng)[0],
                         sequences=[tensor.arange(n_steps)],
                         outputs_info=sample)
    f = function([sample], output, updates=updates) 
    data_2plot= f(mnist.get_data()[0][0:n_samples])
    result = list(itertools.chain(*data_2plot))
    initial_data_point = mnist.get_data()[0][0:n_samples]
    result = numpy.concatenate((initial_data_point, result))
    result = numpy.vstack(result)
    plot_samples(result, n_samples, n_steps)

def plot_samples(result, n_samples, n_steps):
    tiled = image.tile_raster_images(result,
                             img_shape=[28,28],
                             tile_shape=[n_steps, n_samples],
                             tile_spacing=(2,2))
    image.save("binary_RBM_MNIST_example.png", tiled)


if __name__=='__main__':
    visualise_samples(n_samples=10, n_steps=100)
