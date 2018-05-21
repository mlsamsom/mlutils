import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import yaml

from data import load_data, clean_data, train_test_split
from models import LSTMPredictor


CONFIGFILE = "./config/configuration.yaml"

with open(CONFIGFILE, "r") as f:
    config = yaml.load(f)


ap = ArgumentParser()
ap.add_argument('--inspect_data', action='store_true', default=False,
                help="plot training data for inspection")
ap.add_argument('--train', action='store_true', default=False,
                help="Run training")
ap.add_argument('--test', action='store_true', default=False,
                help="Run test")
args = ap.parse_args()

df = load_data()
milk = clean_data(df)
train, test = train_test_split(milk)

if args.inspect_data:
    print("RAW DATA")
    print(df.head())
    milk.plot()
    plt.show()
elif args.train:
    model = LSTMPredictor(config)
    model.fit(train['Milk Production'].values.reshape(1, -1))
    model.close()
elif args.test:
    model = LSTMPredictor(config)
    y_pred = model.infer(train['Milk Production'].values.reshape(1, -1), 12)
    y_pred = list(y_pred)
    test = test.copy()
    test['generated'] = y_pred
    test.plot()
    plt.show()
else:
    print("USAGE:")
    print(" To plot an example of training data")
    print(" $ python main.py --inspect_data")
    print(" To run training")
    print(" $ python main.py --train")
    print(" To test the model")
    print(" $ python main.py --test")
