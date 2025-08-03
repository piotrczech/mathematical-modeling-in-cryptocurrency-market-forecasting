from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from hyperopt import STATUS_OK, fmin, tpe, Trials, space_eval
import json
import time
import keras.backend
import numpy as np
from tensorflow.keras import losses as keras_losses
from rich.table import Table
from rich.console import Console

class HyperparameterOptimizationService:
    def __init__(self, X, y, build_model: callable, loss_function: callable, n_splits: int = 3):
        self.X = X
        self.y = y
        self.build_model = build_model
        self.loss_function = loss_function
        self.n_splits = n_splits

    def objective(self, hparams):
        """Objective function for hyperopt: evaluates hyperparameters using cross-validation with proper scaling."""
        tss = TimeSeriesSplit(n_splits=self.n_splits, gap=1)
        val_losses = []


        print(f'\nEvaluating hyperparameters: {json.dumps(hparams, indent=2)}\n')
        start_time = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(tss.split(self.X)):
            X_tr, X_val = self.X[train_idx], self.X[val_idx]
            y_tr, y_val = self.y[train_idx], self.y[val_idx]

            # Assert shapes
            assert len(X_tr.shape) == 3, f'X_tr must be 3D, got {X_tr.shape}'
            assert len(y_tr.shape) == 2, f'y_tr must be 2D, got {y_tr.shape}'

            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            # Scale X (reshape to 2D for scaler)
            X_tr_resh = X_tr.reshape(-1, X_tr.shape[-1])
            X_val_resh = X_val.reshape(-1, X_val.shape[-1])

            scaler_X.fit(X_tr_resh)
            X_tr_scaled = scaler_X.transform(X_tr_resh).reshape(X_tr.shape)
            X_val_scaled = scaler_X.transform(X_val_resh).reshape(X_val.shape)

            # Scale y
            scaler_y.fit(y_tr)
            y_tr_scaled = scaler_y.transform(y_tr)
            y_val_scaled = scaler_y.transform(y_val)

            model = self.build_model(hparams)

            model.fit(
                X_tr_scaled,
                y_tr_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                epochs=hparams['epochs'],
                batch_size=hparams['batch_size'],
                verbose=0
            )

            y_pred_scaled = model.predict(X_val_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)

            loss_value = self.loss_function(y_val, y_pred).numpy().mean() # Convert tensor to float and take mean

            val_losses.append(loss_value)
            print(f'Fold {fold_idx+1}/{self.n_splits} val_loss: {loss_value}')

            keras.backend.clear_session()  # Clear TensorFlow session to free memory

        mean_loss = np.mean(val_losses)
        print(f'Mean val_loss for this trial: {mean_loss}')
        print(f'Time taken: {time.time() - start_time:.2f} seconds')
        print('\n')

        return {
            'status': STATUS_OK,
            'loss': mean_loss,
        }

    def optimize(self, space: dict, max_evals=70):
        trials = Trials()

        best = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(42)
        )

        best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials])
        best_trial = trials.trials[best_trial_idx]
        
        console = Console()
        console.print("Hyperparameter optimization ended:")
        console.print(f"Best mean loss: {best_trial['result']['loss']:.4f}")

        return space_eval(space, best), trials