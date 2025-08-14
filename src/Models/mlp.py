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

        for i in range(5):
            units = hparams[f'units_layer{i}']
            model.add(keras.layers.Dense(units))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation(hparams['activation']))
            model.add(keras.layers.Dropout(hparams['dropout_rate']))

        model.add(keras.layers.Dense(2, activation=None))

        optimizer = keras.optimizers.get(hparams['optimizer'])
        optimizer.learning_rate = hparams['lr']

        model.compile(optimizer=optimizer, loss=hparams.get('loss_method', 'huber'), metrics=['mae'])

        return model

    def get_space(self, epochs: int, loss_method: str):
        return {
            'epochs': epochs,
            'units_layer0': scope.int(hp.quniform('units_layer0', 256, 1024, 16)),
            'units_layer1': scope.int(hp.quniform('units_layer1', 256, 1024, 16)),
            'units_layer2': scope.int(hp.quniform('units_layer2', 128, 512, 16)),
            'units_layer3': scope.int(hp.quniform('units_layer3', 64, 256, 16)),
            'units_layer4': scope.int(hp.quniform('units_layer4', 64, 256, 16)),
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
            'activation': hp.choice('activation', ['relu', 'elu', 'selu']),
            'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
            'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-2)),
            'batch_size': scope.int(hp.quniform('batch_size', 32, 256, 32)),
            'loss_method': loss_method
        } 