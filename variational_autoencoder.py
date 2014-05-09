from pylearn2.config.yaml_parse import load_path
from mlp import GaussianVariationalCost

train = load_path('MNIST_variational_autoencoder.yaml')

import numpy as np
from theano import function, tensor

x = tensor.matrix()
cost = GaussianVariationalCost()
f = function([x], cost.expr(model, x))
#x_values = np.zeros((2, 10))
print f(data)