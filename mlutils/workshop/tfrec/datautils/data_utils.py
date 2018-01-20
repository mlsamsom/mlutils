import os
import h5py
import numpy as np


DEFAULT_DIR = "/Users/mike/Dropbox/Courses/TensorflowUdemy/ConvNets/assignment/data"


def unpickle(filepath):
    import pickle
    with open(filepath,'rb') as f:
        cifar_dict = pickle.load(f, encoding='bytes')
    return cifar_dict


def one_hot_encode(vec, vals=10):
    """
    For use to one-hot encode the 10- possible labels
    """
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


def cifar_batch_generator(data_dir, batch_names):
    for batch_name in batch_names:
        rawdat = unpickle(os.path.join(data_dir, batch_name))

        X = np.array(rawdat[b'data'])
        y = np.array(rawdat[b'labels'])

        yield X, y


class HDF5Batcher():
    """Create batches from an hdf5 training set

    Set must have datasets organized in <type>_X for cleaned and
    normalized data and <type>_y for one hot encoded labels where
    <type> is train, test or valid.
    """
    def __init__(self, hdffilepath, dataset):
        print("[INFO] opening dataset {}".format(hdffilepath))
        self.h5f = h5py.File(hdffilepath, 'r')
        self.i = 0

    def close(self):
        self.h5f.close()

    def next_batch(self, dataset, batch_size):
        """Returns a data batch of size nex_batches

        Args:
            batch_size : int
        """
        start = self.i
        end = self.i + batch_size
        self.i = end
        X = self.h5f['{}_X'.format(dataset)][start:end]
        y = self.h5f['{}_y'.format(dataset)][start:end]
        return X, y

    def random_batch(self, dataset, batch_size):
        """Grabs a random set of test images
        """
        idx = np.random.randint(0, self._dataset['train_X'].shape[0], size=batch_size)
        X = self.h5f['{}_X'.format(dataset)][idx]
        y = self.h5f['{}_y'.format(dataset)][idx]
        return X, y

    def get_all(self, dataset):
        """Grabs a random set of test images
        """
        X = self.h5f['{}_X'.format(dataset)][:]
        y = self.h5f['{}_y'.format(dataset)][:]
        return X, y


if __name__ == "__main__":
    batcher = HDF5Batcher("../data/data.hdf5", 'train')
    X, y = batcher.next_batch(50)
    print(X.shape)
    print(y.shape)
    print(y[0])
    X, y = batcher.next_batch(50)
    print(X.shape)
    print(y.shape)
    print(y[0])

    batcher.dataset = 'test'
    X, y = batcher.next_batch(50)
    print(X.shape)
    print(y.shape)

    batcher.close()
