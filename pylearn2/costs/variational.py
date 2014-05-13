from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
import numpy as np
from pylearn2.models.mlp import Layer, CompositeLayer
from pylearn2.space import VectorSpace, CompositeSpace
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

class GaussianVariationalCost(DefaultDataSpecsMixin, Cost):
    def expr(self, model, data):
        temp = data
        param_list = []
        for i in xrange(len(model.layers)):
            temp = model.layers[i].fprop(temp)
            if isinstance(model.layers[i], CompositeLayer):
                param_list.append(temp)
        (mean_z, log_sig2_z), (mean_x, log_sig2_x) = param_list
        kl = - .5 * tensor.mean(1 + log_sig2_z - tensor.sqr(mean_z) - tensor.exp(log_sig2_z))
        coef_1 = 1. / tensor.sqrt(tensor.prod(2. * np.pi * tensor.exp(log_sig2_x)))
        prod_2 = tensor.dot(tensor.exp(- log_sig2_x), tensor.transpose(tensor.sqr(data - mean_x))) 
        likl = (coef_1 * tensor.exp(- prod_2)).diagonal()
        return tensor.mean(kl - likl)

class BernoulliVariationalCost(DefaultDataSpecsMixin, Cost):
    def expr(self, model, data):
        temp = data
        param_list = []
        for i in xrange(len(model.layers)):
            temp = model.layers[i].fprop(temp)
            if isinstance(model.layers[i], CompositeLayer):
                param_list.append(temp)
        #import ipdb; ipdb.set_trace()
        (mean_z, log_sig2_z), (bernoulli_x,) = param_list
        kl =  - .5 * tensor.sum(1 + log_sig2_z - tensor.sqr(mean_z) - tensor.exp(log_sig2_z), axis=1)
        #likl = tensor.dot(tensor.transpose(data), tensor.log(bernoulli_x)) + tensor.dot(tensor.transpose(1 - data), tensor.log((1 - bernoulli_x)))
        likl = tensor.sum(data * tensor.log(bernoulli_x) + (1 - data) * tensor.log((1 - bernoulli_x)), axis=1)
        return tensor.mean(kl - likl)
