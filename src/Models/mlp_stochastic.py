import numpy as np
import tensorflow as tf
from tensorflow import keras
from hyperopt import hp
from hyperopt.pyll import scope
from .base_stochastic_model import BaseStochasticModel, StochasticModule


class MLPStochasticModel(BaseStochasticModel):
    def build_model(self, hparams):
        tf.random.set_seed(self.tf_seed)

        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        x = keras.layers.Flatten()(inputs)
        s_t_minus_1 = x

        for i in range(5):
            units = int(hparams[f'units_layer{i}'])
            h_t = keras.layers.Dense(units)(s_t_minus_1)
            h_t = keras.layers.BatchNormalization()(h_t)
            h_t = keras.layers.Activation(hparams['activation'])(h_t)
            s_t = StochasticModule(
                gamma=float(hparams.get('gamma', 0.1)),
                learnable_gamma=bool(hparams.get('learnable_gamma', 0)),
                name=f"stochastic_{i}"
            )([h_t, s_t_minus_1])
            s_t_minus_1 = s_t

        # Warstwa wyjściowa + moduł stochastyczny na końcu
        h_out = keras.layers.Dense(2, activation=None)(s_t_minus_1)
        outputs = StochasticModule(
            gamma=float(hparams.get('gamma', 0.1)),
            learnable_gamma=bool(hparams.get('learnable_gamma', 0)),
            name="stochastic_out"
        )([h_out, s_t_minus_1])

        model = keras.Model(inputs=inputs, outputs=outputs)

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
            # Parametry stochastyczne
            'gamma': hp.uniform('gamma', 0.05, 0.2),
            'learnable_gamma': hp.choice('learnable_gamma', [0, 1]),
            # Pozostałe hiperparametry
            'activation': hp.choice('activation', ['relu', 'elu', 'selu']),
            'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
            'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-2)),
            'batch_size': scope.int(hp.quniform('batch_size', 32, 256, 32)),
            'loss_method': loss_method,
        }


