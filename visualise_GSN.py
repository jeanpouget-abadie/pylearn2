import cPickle
import matplotlib.pyplot as plt

with open('GSN_local.pkl', 'r') as f:
	model = cPickle.load(f)

model.show_examples()
plt.show()

model.show_mc(1, 1000)
plt.show()