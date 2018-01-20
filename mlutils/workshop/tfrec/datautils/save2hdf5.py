import numpy as np
import os
import h5py
from data_utils import cifar_batch_generator, DEFAULT_DIR, one_hot_encode


train_names = ['data_batch_{}'.format(str(i)) for i in range(1, 6)]

train_data = []
train_labels = []
for X, y in cifar_batch_generator(DEFAULT_DIR, train_names):
    train_data.append(X)
    train_labels.append(y)

train_data = np.vstack(train_data)
train_len = len(train_data)
train_data = train_data.reshape(train_len,3,32,32).transpose(0,2,3,1)/255

train_labels = one_hot_encode(np.hstack(train_labels), 10)

print('TRAIN:')
print(train_data.shape)
print(train_labels.shape)

test_data = []
test_labels = []
for X, y in cifar_batch_generator(DEFAULT_DIR, ['test_batch']):
    test_data.append(X)
    test_labels.append(y)

test_data = np.vstack(test_data)
test_len = len(test_data)
test_data = test_data.reshape(test_len,3,32,32).transpose(0,2,3,1)/255
test_labels = one_hot_encode(np.hstack(test_labels), 10)

print("\nTEST")
print(test_data.shape)
print(test_labels.shape)

print("\nSAVING TO HDF5")
h5f = h5py.File(os.path.join(DEFAULT_DIR, 'data.hdf5'), 'w')
h5f.create_dataset('train_X', data=train_data)
h5f.create_dataset('train_y', data=train_labels)
h5f.create_dataset('test_X', data=test_data)
h5f.create_dataset('test_y', data=test_labels)
h5f.close()
