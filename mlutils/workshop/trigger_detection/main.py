import tensorflow as tf
from utils import graph_spectrogram


if __name__ == "__main__":
    x = graph_spectrogram('audio_examples/example_train.wav', show=True)
