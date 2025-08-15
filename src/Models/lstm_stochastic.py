import numpy as np
import tensorflow as tf
from tensorflow import keras
from hyperopt import hp
from hyperopt.pyll import scope
from .base_stochastic_model import BaseStochasticModel, StochasticModule

class LSTMStochasticModel(BaseStochasticModel):
    def build_model(self, hparams):
        tf.random.set_seed(self.tf_seed)

        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        s_t_minus_1 = inputs

        # LSTM layers with stochastic module after each layer
        for i in range(hparams.get('num_lstm_layers', 2)):
            units = int(hparams.get(f'lstm_units_layer{i}', 32))
            return_sequences = (i < hparams.get('num_lstm_layers', 2) - 1) or (hparams.get('global_or_flatten', 'flatten') == 'global')
            lstm_layer = keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
                kernel_regularizer=keras.regularizers.l2(hparams.get('l2_reg', 1e-3)),
                recurrent_dropout=hparams.get('recurrent_dropout', 0.0),
            )
            h = keras.layers.Bidirectional(lstm_layer)(s_t_minus_1) if int(hparams.get('bidirectional', 0)) == 1 else lstm_layer(s_t_minus_1)

            # Ensure compatibility: when return_sequences=False, reduce s_{t-1} to 2D
            if not return_sequences and len(s_t_minus_1.shape) == 3:
                s_t_minus_1_reduced = keras.layers.GlobalAveragePooling1D()(s_t_minus_1)
                s = StochasticModule(
                    gamma=float(hparams.get('gamma', 0.1)),
                    learnable_gamma=bool(hparams.get('learnable_gamma', 0)),
                    name=f"stochastic_{i}"
                )([h, s_t_minus_1_reduced])
            else:
                s = StochasticModule(
                    gamma=float(hparams.get('gamma', 0.1)),
                    learnable_gamma=bool(hparams.get('learnable_gamma', 0)),
                    name=f"stochastic_{i}"
                )([h, s_t_minus_1])
            s_t_minus_1 = s

        if hparams.get('global_or_flatten', 'flatten') == 'global':
            h = keras.layers.GlobalAveragePooling1D()(s_t_minus_1)
        else:
            h = keras.layers.Flatten()(s_t_minus_1)

        h = keras.layers.Dense(int(hparams.get('dense_units', 32)), activation=hparams.get('activation', 'relu'))(h)
        h_out = keras.layers.Dense(2, activation=None)(h)

        outputs = StochasticModule(
            gamma=float(hparams.get('gamma', 0.1)),
            learnable_gamma=bool(hparams.get('learnable_gamma', 0)),
            name="stochastic_out"
        )([h_out, h])

        model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.get(hparams.get('optimizer', 'Adam'))
        optimizer.learning_rate = hparams.get('lr', 1e-3)

        clipnorm = hparams.get('clipnorm', None)
        if clipnorm is not None:
            optimizer.clipnorm = float(clipnorm)

        model.compile(
            optimizer=optimizer,
            loss=hparams.get('loss_method', 'huber'),
            metrics=['mse', 'mae', 'mape']
        )
        return model

    def get_space(self, epochs: int, loss_method: str):
        return {
            # Same as LSTM
            'epochs': epochs,
            'num_lstm_layers': scope.int(hp.quniform('num_lstm_layers', 2, 3, 1)),
            'lstm_units_layer0': scope.int(hp.quniform('lstm_units_layer0', 32, 256, 32)),
            'lstm_units_layer1': scope.int(hp.quniform('lstm_units_layer1', 32, 256, 32)),
            'lstm_units_layer2': scope.int(hp.quniform('lstm_units_layer2', 32, 256, 32)),
            'dense_units': scope.int(hp.quniform('dense_units', 32, 128, 16)),
            'dropout_rate': hp.uniform('dropout_rate', 0.05, 0.3),
            'recurrent_dropout': hp.uniform('recurrent_dropout', 0.0, 0.4),
            'bidirectional': hp.choice('bidirectional', [0, 1]),
            'activation': hp.choice('activation', ['relu', 'elu']),
            'optimizer': hp.choice('optimizer', ['Adam', 'Nadam']),
            'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
            'batch_size': scope.int(hp.quniform('batch_size', 16, 64, 16)),
            'l2_reg': hp.loguniform('l2_reg', np.log(1e-6), np.log(1e-3)),
            'clipnorm': hp.choice('clipnorm', [0.5, 1.0, 2.0]),
            'global_or_flatten': hp.choice('global_or_flatten', ['global', 'flatten']),
            'loss_method': loss_method,

            # Stochastic params
            'gamma': hp.uniform('gamma', 0.001, 0.02),
            'learnable_gamma': hp.choice('learnable_gamma', [0, 1]),
        }


