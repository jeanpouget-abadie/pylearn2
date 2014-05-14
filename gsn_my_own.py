from pylearn2.models.autoencoder import DeepComposedAutoencoder, Autoencoder
from sklearn.datasets import make_moons


data = make_moons(n_samples = 1000, noise = .001)[0]

model = DeepComposedAutoencoder([
			Autoencoder(nvis= , nhid= , act_enc= ,
				act_dec= ),
			Autoencoder(nvis= , nhid= , act_enc= ,
				act_dec= )])
