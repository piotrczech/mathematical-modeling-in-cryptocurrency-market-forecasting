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
        x = inputs

        s_prev = x
        # Stosy LSTM z modułem stochastycznym po każdej warstwie
        for i in range(int(hparams['num_lstm_layers'])):
            units = int(hparams[f'lstm_units_layer{i}'])
            return_sequences = (i < int(hparams['num_lstm_layers']) - 1) or (hparams['global_or_flatten'] == 'global')
            lstm_layer = keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
                kernel_regularizer=keras.regularizers.l2(hparams['l2_reg']),
                recurrent_dropout=hparams.get('recurrent_dropout', 0.0),
            )
            h = keras.layers.Bidirectional(lstm_layer)(s_prev) if int(hparams.get('bidirectional', 0)) == 1 else lstm_layer(s_prev)
            # Zapewnij zgodność rang: gdy return_sequences=False, redukuj s_prev do 2D
            if not return_sequences and len(s_prev.shape) == 3:
                s_prev_reduced = keras.layers.GlobalAveragePooling1D()(s_prev)
                s = StochasticModule(
                    gamma=float(hparams.get('gamma', 0.1)),
                    learnable_gamma=bool(hparams.get('learnable_gamma', 0)),
                    name=f"stochastic_{i}"
                )([h, s_prev_reduced])
            else:
                s = StochasticModule(
                    gamma=float(hparams.get('gamma', 0.1)),
                    learnable_gamma=bool(hparams.get('learnable_gamma', 0)),
                    name=f"stochastic_{i}"
                )([h, s_prev])
            s_prev = s

        if hparams['global_or_flatten'] == 'global':
            h = keras.layers.GlobalAveragePooling1D()(s_prev)
        else:
            h = keras.layers.Flatten()(s_prev)

        h = keras.layers.Dense(int(hparams['dense_units']), activation=hparams['activation'])(h)
        h_out = keras.layers.Dense(2, activation=None)(h)
        # Końcowy moduł stochastyczny: użyj zredukowanego kontekstu `h` (nie sekwencji `s_prev`),
        # aby dopasować rangi i uniknąć kolizji kształtów [B,2] vs [B,T,2]
        outputs = StochasticModule(
            gamma=float(hparams.get('gamma', 0.1)),
            learnable_gamma=bool(hparams.get('learnable_gamma', 0)),
            name="stochastic_out"
        )([h_out, h])

        model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.get(hparams['optimizer'])
        optimizer.learning_rate = hparams['lr']
        clipnorm = hparams.get('clipnorm', None)
        if clipnorm is not None:
            optimizer.clipnorm = float(clipnorm)
        loss_method = hparams.get('loss_method', 'huber')
        model.compile(optimizer=optimizer, loss=loss_method, metrics=['mae'])
        return model

    def get_space(self, epochs: int, loss_method: str):
        return {
            'epochs': epochs,
            'num_lstm_layers': scope.int(hp.quniform('num_lstm_layers', 2, 4, 1)),
            'lstm_units_layer0': scope.int(hp.quniform('lstm_units_layer0', 32, 256, 32)),
            'lstm_units_layer1': scope.int(hp.quniform('lstm_units_layer1', 32, 256, 32)),
            'lstm_units_layer2': scope.int(hp.quniform('lstm_units_layer2', 32, 256, 32)),
            'lstm_units_layer3': scope.int(hp.quniform('lstm_units_layer3', 32, 256, 32)),
            'lstm_units_layer4': scope.int(hp.quniform('lstm_units_layer4', 32, 256, 32)),
            'dense_units': scope.int(hp.quniform('dense_units', 32, 128, 16)),
            'recurrent_dropout': hp.uniform('recurrent_dropout', 0.0, 0.4),
            'bidirectional': hp.choice('bidirectional', [0, 1]),
            'activation': hp.choice('activation', ['relu', 'elu']),
            'optimizer': hp.choice('optimizer', ['Adam', 'Nadam']),
            'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
            'batch_size': scope.int(hp.quniform('batch_size', 16, 64, 16)),
            'l2_reg': hp.loguniform('l2_reg', np.log(1e-6), np.log(1e-3)),
            'clipnorm': hp.choice('clipnorm', [0.5, 1.0, 2.0]),
            'global_or_flatten': hp.choice('global_or_flatten', ['global', 'flatten']),
            # Parametry stochastyczne
            'gamma': hp.uniform('gamma', 0.05, 0.2),
            'learnable_gamma': hp.choice('learnable_gamma', [0, 1]),
            'loss_method': loss_method,
        }


