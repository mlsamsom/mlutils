from utils import graph_spectrogram
from model import GRUModel


Tx = 5511
Ty = 1375
n_freq = 101


if __name__ == "__main__":
    m = GRUModel("./config.yaml")
