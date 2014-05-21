"""
.. todo::

    WRITEME
"""
import numpy as np
from theano import tensor
from theano.compat.python2x import OrderedDict
import theano.sparse
from theano.tensor.shared_randomstreams import RandomStreams

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.utils import sharedX
from pylearn2.datasets.transformer_dataset import TransformerDataset

# class DivergenceCost(DefaultDataSpecsMixin, Cost):
#     def __init__(self, X, num_samples, num_encodings):
#         if not isinstance(X.get_data(), tuple):
#             X = sharedX(X.get_data())
#         elif X.get_data()[1] == 1:
#             print "implementing neighbors solution"
#             X = sharedX(X.get_data()[0])
#             neigh_indices = sharedX(X.get_data()[1])
#         else:
#             print "Original data set is tuple (possible labels) : Warning : (cost/autoencoder.py l22)"
#             X = sharedX(X.get_data()[0])
#         self.__dict__.update(locals())
#         self.rng = RandomStreams(1432)
#         del self.self

#     def sample(self, X, num_X, num_samples):
#         """
#         Parameters
#         ----------
#         X : 2d-array
#         num_X : number of times to sample
#         num_samples : number of samples to take

#         Returns
#         -------
#         output : 3d-array
#             With axes [datapoint, samples, data]
#         """
#         sample_indices = self.rng.permutation(
#             size=(num_X,),
#             n=self.X.shape[0]
#         )[:, :num_samples]
#         return X[sample_indices]

#     def expr(self, model, data):
#         samples = self.sample(self.X, data.shape[0], self.num_samples)
#         flattened_samples = \
#             samples.dimshuffle(2, 0, 1).flatten(samples.ndim - 1).T
#         outputs = model.reconstruct(flattened_samples)
#         outputs = outputs.reshape((samples.shape[0],
#                                    samples.shape[1],
#                                    outputs.shape[1]))
#         likelihood = self.likelihood(data, outputs,
#                                      model.decorruptor.shared_stdev)
#         cost = -tensor.log(likelihood.mean(axis=1)).mean()
#         return cost

#     def likelihood(self, data, outputs, stdev):
#         """
#         Parameters
#         ----------
#         data : 2-d array [batch, data]
#         outputs: 3-d array [batch, samples, data]
#         stdev : shared variable

#         Returns
#         -------
#         likelihood : 2-d array [batch, samples]
#         """
#         distances = (data[:, None, :] - outputs).norm(2, axis=-1) ** 2
#         exponent = -distances / (2. * stdev ** 2.)
#         return 1. / tensor.sqrt(2 * np.pi * stdev ** 2) * tensor.exp(exponent)

#     def get_monitoring_channels(self, model, data):
#         channels = OrderedDict()
#         channels['cost'] = self.expr(model, data)
#         channels['enc_stdev'] = model.corruptor.shared_stdev
#         channels['dec_stdev'] = model.decorruptor.shared_stdev
#         # channels.update(model.act_enc.get_monitoring_channels((data, None)))
#         # channels.update(model.act_dec.get_monitoring_channels((data, None)))
#         return channels


class DivergenceCost_local(DefaultDataSpecsMixin, Cost):
    def __init__(self, X, num_encodings):
        # if isinstance(X, TransformerDataset):
        #     print "implementing transformer_dataset_solution"
        #     X = X.transformer.reconstruct(X.raw.get_data())
        # else:
            # if not isinstance(X.get_data(), tuple):
                # X = sharedX(X.get_data())
            # elif X.get_data()[1] == 1:
            #     print "implementing neighbors solution"
            #     X = sharedX(X.get_data()[0])
            #     neigh_indices = sharedX(X.get_data()[1])
            # else:
            #     print "Original data set is tuple (possible labels) : Warning : (cost/autoencoder.py l22)"
            #     X = sharedX(X.get_data()[0])
        X = sharedX(X.get_data())
        self.__dict__.update(locals())
        self.rng = RandomStreams(1432)
        del self.self

    def expr(self, model, data):
        samples = self.X.dimshuffle(0, 'x', 1)
        corr_samples = model.corrupt(samples, shape=(self.X.shape[0], self.num_encodings, self.X.shape[1]))
        flattened_samples = corr_samples.dimshuffle(2, 0, 1).flatten(corr_samples.ndim - 1).T
        outputs = model.decode_predecorr(flattened_samples)
        outputs = outputs.reshape((self.X.shape[0],
                                   self.num_encodings,
                                   self.X.shape[1]))
        likelihood = self.likelihood(model, data, outputs,
                                     model.decorruptor.shared_stdev)
        cost = -tensor.log(likelihood.mean(axis=-1).mean(axis=-2)).mean(axis=0)
        return cost

    def likelihood(self, model, data, outputs, stdev):
        distances = tensor.sqr(data[:, None, None, :] - (outputs[None, :, :, :])).sum(axis=-1)
        exponent = -distances / (2. * stdev ** 2.)
        return 1. / tensor.sqrt(2 * np.pi * (stdev ** 2)) * tensor.exp(exponent)

    def get_monitoring_channels(self, model, data):
        channels = OrderedDict()
        channels['enc_stdev'] = model.corruptor.shared_stdev
        channels['dec_stdev'] = model.decorruptor.shared_stdev
        return channels


class GSNFriendlyCost(DefaultDataSpecsMixin, Cost):
    """
    .. todo::

        WRITEME
    """

    @staticmethod
    def cost(target, output):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError

    def expr(self, model, data, *args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        X = data
        return self.cost(X, model.reconstruct(X))


class MeanSquaredReconstructionError(GSNFriendlyCost):
    """
    .. todo::

        WRITEME
    """

    @staticmethod
    def cost(a, b):
        """
        .. todo::

            WRITEME
        """
        return ((a - b) ** 2).sum(axis=1).mean()

class MeanBinaryCrossEntropy(GSNFriendlyCost):
    """
    .. todo::

        WRITEME
    """

    @staticmethod
    def cost(target, output):
        """
        .. todo::

            WRITEME
        """
        return tensor.nnet.binary_crossentropy(output, target).sum(axis=1).mean()

class SampledMeanBinaryCrossEntropy(DefaultDataSpecsMixin, Cost):
    """
    .. todo::

        WRITEME properly

    CE cost that goes with sparse autoencoder with L1 regularization on activations

    For theory:
    Y. Dauphin, X. Glorot, Y. Bengio. ICML2011
    Large-Scale Learning of Embeddings with Reconstruction Sampling

    Parameters
    ----------
    L1 : WRITEME
    ratio : WRITEME
    """

    def __init__(self, L1, ratio):
        self.random_stream = RandomStreams(seed=1)
        self.L1 = L1
        self.one_ratio = ratio

    def expr(self, model, data, ** kwargs):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        X = data
        # X is theano sparse
        X_dense = theano.sparse.dense_from_sparse(X)
        noise = self.random_stream.binomial(size=X_dense.shape, n=1,
                                            prob=self.one_ratio, ndim=None)

        # a random pattern that indicates to reconstruct all the 1s and some of the 0s in X
        P = noise + X_dense
        P = theano.tensor.switch(P>0, 1, 0)
        P = tensor.cast(P, theano.config.floatX)

        # L1 penalty on activations
        reg_units = theano.tensor.abs_(model.encode(X)).sum(axis=1).mean()

        # penalty on weights, optional
        # params = model.get_params()
        # W = params[2]

        # there is a numerical problem when using
        # tensor.log(1 - model.reconstruct(X, P))
        # Pascal fixed it.
        before_activation = model.reconstruct_without_dec_acti(X, P)

        cost = ( 1 * X_dense *
                 tensor.log(tensor.log(1 + tensor.exp(-1 * before_activation))) +
                 (1 - X_dense) *
                 tensor.log(1 + tensor.log(1 + tensor.exp(before_activation)))
               )

        cost = (cost * P).sum(axis=1).mean()

        cost = cost + self.L1 * reg_units

        return cost



class SampledMeanSquaredReconstructionError(MeanSquaredReconstructionError):
    """
    mse cost that goes with sparse autoencoder with L1 regularization on activations

    For theory:
    Y. Dauphin, X. Glorot, Y. Bengio. ICML2011
    Large-Scale Learning of Embeddings with Reconstruction Sampling

    Parameters
    ----------
    L1 : WRITEME
    ratio : WRITEME
    """

    def __init__(self, L1, ratio):
        self.random_stream = RandomStreams(seed=1)
        self.L1 = L1
        self.ratio = ratio

    def expr(self, model, data, ** kwargs):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        X = data
        # X is theano sparse
        X_dense=theano.sparse.dense_from_sparse(X)
        noise = self.random_stream.binomial(size=X_dense.shape, n=1, prob=self.ratio, ndim=None)

        # a random pattern that indicates to reconstruct all the 1s and some of the 0s in X
        P = noise + X_dense
        P = theano.tensor.switch(P>0, 1, 0)
        P = tensor.cast(P, theano.config.floatX)

        # L1 penalty on activations
        L1_units = theano.tensor.abs_(model.encode(X)).sum(axis=1).mean()

        # penalty on weights, optional
        #params = model.get_params()
        #W = params[2]
        #L1_weights = theano.tensor.abs_(W).sum()

        cost = ((model.reconstruct(X, P) - X_dense) ** 2)

        cost = (cost * P).sum(axis=1).mean()

        cost = cost + self.L1 * L1_units

        return cost


#class MeanBinaryCrossEntropyTanh(Cost):
#     def expr(self, model, data):
#        self.get_data_specs(model)[0].validate(data)
#        X = data
#        X = (X + 1) / 2.
#        return (
#            tensor.xlogx.xlogx(model.reconstruct(X)) +
#            tensor.xlogx.xlogx(1 - model.reconstruct(X))
#        ).sum(axis=1).mean()
#
#    def get_data_specs(self, model):
#        return (model.get_input_space(), model.get_input_source())


class SparseActivation(DefaultDataSpecsMixin, Cost):
    """
    Autoencoder sparse activation cost.
    
    Regularize on KL divergence from desired average activation of each
    hidden unit as described in Andrew Ng's CS294A Lecture Notes. See
    http://www.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf.

    Parameters
    ----------
    coeff : float
        Coefficient for this regularization term in the objective
        function.
    p : float
        Desired average activation of each hidden unit.
    """
    def __init__(self, coeff, p):
        self.coeff = coeff
        self.p = p

    def expr(self, model, data, **kwargs):
        """
        Calculate regularization penalty.
        """
        X = data
        p = self.p
        p_hat = tensor.abs_(model.encode(X)).mean(axis=0)
        kl = p * tensor.log(p / p_hat) + (1 - p) * \
            tensor.log((1 - p) / (1 - p_hat))
        penalty = self.coeff * kl.sum()
        penalty.name = 'sparse_activation_penalty'
        return penalty
