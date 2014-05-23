import cPickle
import matplotlib.pyplot as plt
from pylearn2.datasets.mnist import MNIST

def twodsets():
    with open('GSN_2_moons_10hid_2enc.pkl', 'r') as f:
        model = cPickle.load(f)

    model.show_examples()
    plt.show()

    model.show_mc(1, 1000)
    plt.show()
    #plt.savefig('good_GSN_local_10hid_34000epoch_100spl_10enc_sigm.png')

def MNISTsets():
    with open('GSN_MNIST.pkl') as f:
        model = cPickle.load(f)
    
    mb_data = MNIST(which_set='test').X[106:107, :]
    model.show_mc(initial_data_point=mb_data, n_steps=1)


if __name__=='__main__':
    MNISTsets()
    #twodsets()