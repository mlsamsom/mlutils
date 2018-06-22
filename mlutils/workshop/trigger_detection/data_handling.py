# script to create training set

import yaml
import os
from utils import load_raw_audio, create_training_example

Ty = 1375

with open("./config.yaml", "r") as f:
    config = yaml.load(f)

dat_dir = os.path.abspath(config['data_dir'])
train_dir = os.path.abspath(config['train_dir'])

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

act, neg, bkgrd = load_raw_audio(dat_dir)

nm = os.path.join(train_dir, "train_{}.wav".format(0))
x, y = create_training_example(bkgrd[0], act, neg, nm, Ty, show=True)
print(x.shape)
print(y.shape)
