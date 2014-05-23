import cPickle as pickle
import itertools

import numpy as np
import theano.tensor as T

from pylearn2.expr.activations import rescaled_softmax
from pylearn2.costs.autoencoder import MeanBinaryCrossEntropy
from pylearn2.costs.gsn import GSNCost
from pylearn2.corruption import (BinomialSampler, GaussianCorruptor,
                                 MultinomialSampler, SaltPepperCorruptor,
                                 SmoothOneHotCorruptor)
from pylearn2.datasets.mnist import MNIST
from pylearn2.distributions.parzen import ParzenWindows
from pylearn2.models.gsn import GSN, JointGSN
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD, MonitorBasedLRAdjuster
from pylearn2.utils import image, safe_zip

# define some common parameters
HIDDEN_SIZE = 1000
SALT_PEPPER_NOISE = 0.4
GAUSSIAN_NOISE = 0.5

WALKBACK = 0

LEARNING_RATE = 0.25
MOMENTUM = 0.75

MAX_EPOCHS = 100
BATCHES_PER_EPOCH = None # covers full training set
BATCH_SIZE = 100

ds = MNIST(which_set='train')

def test_train_ae():
    GC = GaussianCorruptor

    gsn = GSN.new(
        layer_sizes=[ds.X.shape[1], 1000],
        activation_funcs=["sigmoid", "tanh"],
        pre_corruptors=[None, GC(1.0)],
        post_corruptors=[SaltPepperCorruptor(0.5), GC(1.0)],
        layer_samplers=[BinomialSampler(), None],
        tied=False
    )

    ##################
    # average MBCE over example rather than sum it
    _mbce = MeanBinaryCrossEntropy()
    reconstruction_cost = lambda a, b: _mbce.cost(a, b) / ds.X.shape[1]

    c = GSNCost([(0, 1.0, reconstruction_cost)], walkback=WALKBACK)
    ###################

    alg = SGD(
        LEARNING_RATE,
        init_momentum=MOMENTUM,
        cost=c,
        termination_criterion=EpochCounter(MAX_EPOCHS),
        batches_per_iter=BATCHES_PER_EPOCH,
        batch_size=BATCH_SIZE,
        monitoring_dataset=ds,
        monitoring_batches=10
   )

    trainer = Train(ds, gsn, algorithm=alg, save_path="gsn_ae_example.pkl",
                    save_freq=5)
    trainer.main_loop()
    print "done training"



def test_sample_ae():
    """
    Visualize some samples from the trained unsupervised GSN.
    """
    with open("gsn_ae_example.pkl") as f:
        gsn = pickle.load(f)

    # random point to start at
    mb_data = MNIST(which_set='test').X[106:107, :]

    history = gsn.get_samples([(0, mb_data)], walkback=500,
                              symbolic=False, include_first=True)

   
    history = list(itertools.chain(*history))
    history = np.vstack(history)

    tiled = image.tile_raster_images(history,
                                     img_shape=[28,28],
                                     tile_shape=[50,50],
                                     tile_spacing=(2,2))
    image.save("gsn_ae_example.png", tiled)

    # code to get log likelihood from kernel density estimator
    # this crashed on GPU (out of memory), but works on CPU
    pw = ParzenWindows(MNIST(which_set='test').X, .20)
    print pw.get_ll(history)

if __name__=="__main__":
    #test_train_ae()
    test_sample_ae()