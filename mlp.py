from pylearn2.costs.cost import Cost
import numpy as np
from pylearn2.models.mlp import Layer, CompositeLayer
from pylearn2.space import VectorSpace, CompositeSpace
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

class SamplingLayer(Layer):
    def __init__(self, layer_name):
        super(SamplingLayer, self).__init__()
        self.layer_name = layer_name

    def fprop(self, state_below):
        return state_below[0] + tensor.exp(state_below[1] / 2.) * self.mlp.rng.normal()

    def set_input_space(self, space):
        assert isinstance(space, CompositeSpace)
        assert len(space.components) == 2
        assert space.components[0] == space.components[1]

        self.output_space = VectorSpace(dim=space.components[0].dim)

class VariationalCost(Cost):
    def __init__(self):
        pass

    def expr(self, model, data):
        temp = data
        param_list = []
        for i in xrange(len(model.layers)):
            temp = model.layers[i].fprop(temp)
            if isinstance(model.layers[i], CompositeLayer):
                param_list.append(temp)
        (mean_z, log_sig2_z), (mean_x, log_sig2_x) = param_list
        kl = .5 * tensor.sum(1 + log_sig2_z - tensor.sqr(mean_z) - tensor.exp(log_sig2_z))
        coef_1 = 1. / tensor.sqrt(tensor.prod(2. * np.pi * tensor.exp(log_sig2_x)))
        prod_2 = tensor.dot(tensor.exp(- log_sig2_x), tensor.transpose(tensor.sqr(data - mean_x))) 
        likl = (coef_1 * tensor.exp(- prod_2)).diagonal()
        return kl - likl
