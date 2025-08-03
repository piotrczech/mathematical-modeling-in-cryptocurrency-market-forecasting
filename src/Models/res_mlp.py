import numpy as np
import tensorflow as tf
from tensorflow import keras
from hyperopt import hp
from hyperopt.pyll import scope
from .base_model import BaseModel
import tensorflow.keras.regularizers as regularizers

class ResMLPModel(BaseModel):
    def build_model(self, hparams):
        tf.random.set_seed(self.tf_seed)
        
        input_layer = keras.Input(shape=(self.sequence_length, self.n_features))
        
        # Redukcja sekwencji na podstawie pooling_type
        if hparams['pooling_type'] == 'global_avg':
            x = keras.layers.GlobalAveragePooling1D()(input_layer)
        elif hparams['pooling_type'] == 'flatten':
            x = keras.layers.Flatten()(input_layer)

        # Główna pętla: residual dense blocks
        for i in range(hparams['num_blocks']):
            x = self.residual_block(
                x,
                units=hparams[f'units_block{i}'],
                activation=hparams['activation'],
                dropout_rate=hparams['dropout_rate'],
                layers_per_block=hparams['layers_per_block'],
                l2_reg=hparams['l2_reg']
            )

        # Wyjście: 2 wartości regresji
        output_layer = keras.layers.Dense(2)(x)

        model = keras.Model(inputs=input_layer, outputs=output_layer)
        
        optimizer = keras.optimizers.get(hparams['optimizer'])
        optimizer.learning_rate = hparams['lr']

        model.compile(
            optimizer=optimizer,
            loss=hparams.get('loss_method', 'huber'),
            metrics=['mae']
        )

        return model

    def residual_block(self, x, units, activation='relu', dropout_rate=0.2, layers_per_block=1, l2_reg=0.0):
        shortcut = x
        for _ in range(layers_per_block):
            x = keras.layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activation)(x)
            x = keras.layers.Dropout(dropout_rate)(x)

        # Dopasuj shortcut jeśli trzeba (np. inny wymiar)
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = keras.layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg))(shortcut)
        
        x = keras.layers.Add()([x, shortcut])
        return x

    def get_space(self, epochs: int, loss_method: str):
        space = {
            'epochs': epochs,
            'num_blocks': scope.int(hp.quniform('num_blocks', 2, 4, 1)),  # liczba bloków res
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.4),
            'activation': hp.choice('activation', ['relu', 'elu', 'selu']),
            'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
            'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-2)),
            'batch_size': scope.int(hp.quniform('batch_size', 32, 256, 32)),
            'loss_method': loss_method
        }

        # Jednostki dla każdego potencjalnego bloku
        for i in range(4):  # max bloków
            space[f'units_block{i}'] = scope.int(hp.quniform(f'units_block{i}', 64, 512, 32))

        space['pooling_type'] = hp.choice('pooling_type', ['global_avg', 'flatten'])
        space['layers_per_block'] = scope.int(hp.quniform('layers_per_block', 1, 3, 1))
        space['l2_reg'] = hp.loguniform('l2_reg', np.log(1e-5), np.log(1e-2))

        return space
