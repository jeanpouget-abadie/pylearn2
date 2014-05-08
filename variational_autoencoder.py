from pylearn2.config.yaml_parse import load_path
from mlp import VariationalCost

model = load_path('config_variational_autoencoder.yaml')

# x_dim = mlp_p.get_output_space().get_total_dimension()
# z_dim = mlp_q.get_output_space().get_total_dimension()
# print mlp_p.fprop()

import numpy as np
from theano import function, tensor

x = tensor.matrix()
cost = VariationalCost()
f = function([x], cost.expr(model, x))
x_values = np.zeros((2, 10))
print f(x_values)