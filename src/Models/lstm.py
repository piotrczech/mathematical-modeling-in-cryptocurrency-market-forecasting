import numpy as np
import tensorflow as tf
from tensorflow import keras
from hyperopt import hp
from hyperopt.pyll import scope
from .base_model import BaseModel

class LSTMModel(BaseModel):
    def build_model(self, hparams):
        tf.random.set_seed(self.tf_seed)

        model = keras.models.Sequential()
        model.add(keras.Input(shape=(self.sequence_length, self.n_features)))

        for i in range(hparams['num_lstm_layers']):
            units = int(hparams[f'lstm_units_layer{i}'])
            return_sequences = (i < hparams['num_lstm_layers'] - 1)
            model.add(keras.layers.LSTM(units, return_sequences=return_sequences, kernel_regularizer=keras.regularizers.l2(hparams['l2_reg'])))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(hparams['dropout_rate']))
    
        model.add(keras.layers.Flatten())  # Flatten after LSTM if return_sequences=False
        model.add(keras.layers.Dense(int(hparams['dense_units']), activation=hparams['activation']))
        model.add(keras.layers.Dense(2, activation=hparams['activation']))  # Low and High for each day
        
        optimizer = keras.optimizers.get(hparams['optimizer'])
        optimizer.learning_rate = hparams['lr']
        
        loss_method = hparams.get('loss_method', 'huber')
        model.compile(optimizer=optimizer, loss=loss_method, metrics=['mae'])
    
        return model

    def get_space(self, epochs: int, loss_method: str):
        return {
            'epochs': epochs,
            'num_lstm_layers': 3,
            'lstm_units_layer0': scope.int(hp.quniform('lstm_units_layer0', 32, 128, 32)),
            'lstm_units_layer1': scope.int(hp.quniform('lstm_units_layer1', 32, 128, 32)),
            'lstm_units_layer2': scope.int(hp.quniform('lstm_units_layer2', 32, 128, 32)),
            'dense_units': scope.int(hp.quniform('dense_units', 16, 64, 16)),
            'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.5),
            'activation': hp.choice('activation', ['relu', 'elu']),
            'optimizer': hp.choice('optimizer', ['Adam', 'Nadam']),
            'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
            'batch_size': scope.int(hp.quniform('batch_size', 16, 64, 16)),
            'l2_reg': hp.loguniform('l2_reg', np.log(1e-5), np.log(1e-2)),
            'loss_method': loss_method
        } 