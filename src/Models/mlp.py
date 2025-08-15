import numpy as np
import tensorflow as tf
from tensorflow import keras
from hyperopt import hp
from hyperopt.pyll import scope
from .base_model import BaseModel

class MLPModel(BaseModel):
    def build_model(self, hparams):
        tf.random.set_seed(self.tf_seed)

        model = keras.Sequential()
        model.add(keras.Input(shape=(self.sequence_length, self.n_features)))
        model.add(keras.layers.Flatten())

        for i in range(hparams.get('num_layers', 2)):
            units = int(hparams.get(f'units_layer{i}', 64))
            model.add(keras.layers.Dense(units))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation(hparams.get('activation', 'relu')))
            model.add(keras.layers.Dropout(hparams.get('dropout_rate', 0.1)))

        model.add(keras.layers.Dense(2, activation=None))

        optimizer = keras.optimizers.get(hparams.get('optimizer', 'Adam'))
        optimizer.learning_rate = hparams.get('lr', 1e-3)

        model.compile(
            optimizer=optimizer,
            loss=hparams.get('loss_method', 'huber'),
            metrics=['mse', 'mae', 'mape']
        )

        return model

    def get_space(self, epochs: int, loss_method: str):
        return {
            'epochs': epochs,
            'num_layers': scope.int(hp.quniform('num_layers', 2, 4, 1)),
            'units_layer0': scope.int(hp.quniform('units_layer0', 64, 512, 16)),
            'units_layer1': scope.int(hp.quniform('units_layer1', 64, 512, 16)),
            'units_layer2': scope.int(hp.quniform('units_layer2', 64, 256, 16)),
            'units_layer3': scope.int(hp.quniform('units_layer3', 64, 256, 16)),
            'dropout_rate': hp.uniform('dropout_rate', 0.05, 0.3),
            'activation': hp.choice('activation', ['relu', 'elu', 'selu']),
            'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
            'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-2)),
            'batch_size': scope.int(hp.quniform('batch_size', 16, 64, 32)),
            'loss_method': loss_method
        } 