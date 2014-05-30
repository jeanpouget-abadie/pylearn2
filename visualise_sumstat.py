import cPickle
from pylearn2.datasets.mnist import MNIST
from pylearn2.utils.rng import make_theano_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import function, tensor, scan
from pylearn2.utils import image
import itertools
import numpy
import PIL


def sample_from_chain(model, x, rng):
    return model.gibbs_step_for_v(x, rng)[0]


def import_model():
    #with open('binary_RBM_sumstat_puerhexperiment.pkl', 'r') as f:
    with open('binary_RBM_sumstat.pkl', 'r') as f:
        model = cPickle.load(f)
    mnist = MNIST('train')
    rng = RandomStreams(123)
    return model, rng, mnist


def visualise_samples_from_v(n_samples, n_steps, which_model):
    model, rng, mnist = import_model()
    assert isinstance(which_model, int)  # 0 : Recognition # 1 : Generative
    model = model.rbms[which_model]

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


def visualise_samples_from_h_simple(n_samples, which_model):
    model, rng, mnist = import_model()
    assert isinstance(which_model, int)  # 0 : Recognition # 1 : Generative
    model = model.rbms[which_model]

    sample = tensor.fmatrix()
    output = model.sample_visibles(params=model.mean_v_given_h(sample),
                                   shape=mnist.get_data()[0][0:n_samples].shape,
                                   rng=rng)
    f = function([sample], output)

    prior_z = numpy.random.randint(2, size=[n_samples, 200]).astype('float32')

    data_2plot = f(prior_z)
    plot_samples(data_2plot, n_samples, 1)


# def visualise_samples_from_h(n_samples, n_steps, which_model):
#     model, rng, mnist = import_model()
#     assert isinstance(which_model, int)  # 0 : Recognition # 1 : Generative
#     model = model.rbms[which_model]

#     sample = tensor.fmatrix()
#     ####
#     #things need to be changed
#     ####
#     output, updates = scan(fn=lambda i, x_init: model.sample_visibles(
#                             model.mean_v_given_h(), shape, rng),
#                          sequences=[tensor.arange(n_steps)],
#                          outputs_info=sample)
#     f = function([sample], output, updates=updates)

#     data_2plot= f(mnist.get_data()[0][0:n_samples])

#     result = list(itertools.chain(*data_2plot))
#     initial_data_point = mnist.get_data()[0][0:n_samples]
#     result = numpy.concatenate((initial_data_point, result))
#     result = numpy.vstack(result)

#     plot_samples(result, n_samples, n_steps)


def look_at_filters(which_model):
    model, rng, mnist = import_model()
    assert isinstance(which_model, int)  # 0 : Recognition # 1 : Generative
    rbm = model.rbms[which_model]

    tiled = PIL.Image.fromarray(image.tile_raster_images(
             X=rbm.get_weights(borrow=True).T,
             img_shape=(28, 28), tile_shape=(10, 10),
             tile_spacing=(1, 1)))
    image.save('filter.png', tiled)


def plot_samples(result, n_samples, n_steps):
    tiled = image.tile_raster_images(result,
                             img_shape=[28,28],
                             tile_shape=[n_steps, n_samples],
                             tile_spacing=(2,2))
    image.save("binary_RBM_MNIST_example_puerhexample.png", tiled)


if __name__=='__main__':
    visualise_samples_from_v(n_samples=10, n_steps=100, which_model=1)
    #visualise_samples_from_h_simple(n_samples=10, which_model=1)
    #look_at_filters(0)
