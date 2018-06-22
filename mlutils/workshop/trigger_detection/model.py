from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam


class GRUModel(object):
    """Simple 2 layer GRU Model
    """
    def __init__(self, config):
        # unpack config
        pass

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

    def _init(self):
        pass

    def save(self):
        pass
