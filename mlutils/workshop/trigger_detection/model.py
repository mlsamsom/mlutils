from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from datetime import datetime
import yaml
import os


class GRUModel(object):
    """Simple 2 layer GRU Model
    """
    def __init__(self, config):
        config = os.path.abspath(config)
        # unpack config
        with open(config, "r") as f:
            config_dict = yaml.load(f)
        self._result_dir = os.path.abspath(config_dict["result_dir"])
        self._init()

    def define_model(self):
        """Define word detection model
        """
        X_input = Input(shape=self._input_shape)

        # CONV layer
        X = Conv1D(196, 15, strides=4)(X_input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Dropout(0.8)(X)

        # First GRU Layer
        X = GRU(units = 128, return_sequences=True)(X)
        X = Dropout(0.8)(X)
        X = BatchNormalization()(X)

        # Second GRU Layer
        X = GRU(units = 128, return_sequences=True)(X)
        X = Dropout(0.8)(X)
        X = BatchNormalization()(X)
        X = Dropout(0.8)(X)

        # Time-distributed dense layer
        X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)

        model = Model(inputs=X_input, outputs=X)

        return model

    def fit(self, X, y, batch_size=5, epochs=5, learning_rate=1e-4):
        """Fits the model
        """
        opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.01)
        self._model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=["accuracy"]
        )
        self._model.fit(X, y, batch_size=batch_size, epochs=epochs)
        self.save()

    def eval(self, X, y):
        """Run evaluation
        """
        return self._model.evaluate(X, y)

    def _init(self):
        """Initialize model
        """
        if not os.path.exists(self._result_dir):
            print("No saved models in {}".format(self._result_dir))
            os.mkdir(self._result_dir)
            return

        modelnames = os.listdir(self._result_dir)
        modelnames = [x for x in modelnames if x.endswith('.hdf5')]
        if len(modelnames) == 0:
            print("No saved models in {}".format(self._result_dir))
            return

        modelnames.sort(reverse=True)
        model_file = os.path.join(self._result_dir, modelnames[0])
        self._model.load_weights(model_file)

    def save(self):
        """Save the model
        """
        nm = datetime.now().strftime("%Y-%m-%d_GRUModel.hdf5")
        sv = os.path.join(self._result_dir, nm)
        self._model.save_weights(sv)
