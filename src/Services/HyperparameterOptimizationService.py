from sklearn.model_selection import TimeSeriesSplit
from hyperopt import STATUS_OK, fmin, tpe, Trials, space_eval
import json
import time
import keras.backend
import numpy as np
from tensorflow.keras import losses as keras_losses
from rich.table import Table
from rich.console import Console
from tensorflow import keras
import tensorflow as tf
from src.Preprocessing import RobustFeatureScaler, TargetScaler

class HyperparameterOptimizationService:
    def __init__(self, X, y, build_model: callable, loss_function: callable, n_splits: int = 3):
        # Ensure consistent dtype to reduce retracing and unnecessary casts
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
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

            # Bez wycieku: osobna instancja skalera per fold, fit tylko na train
            x_scaler = RobustFeatureScaler().fit(X_tr)
            X_tr = x_scaler.transform(X_tr)
            X_val = x_scaler.transform(X_val)

            # Skaluj także y (per fold, bez wycieku)
            y_scaler = TargetScaler().fit(y_tr)
            y_tr = y_scaler.transform(y_tr)
            y_val = y_scaler.transform(y_val)

            model = self.build_model(hparams)

            # Train
            callbacks = [
                keras.callbacks.EarlyStopping(patience=round(hparams['epochs']/4), restore_best_weights=True, monitor='val_loss'),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=round(hparams['epochs']/4), monitor='val_loss'),
            ]
            model.fit(
                X_tr,
                y_tr,
                validation_data=(X_val, y_val),
                epochs=hparams['epochs'],
                batch_size=hparams['batch_size'],
                shuffle=False,
                callbacks=callbacks,
                verbose=0
            )

            # Evaluate with fixed batch size to stabilize function signatures
            # Optionally use tf.data with drop_remainder to keep shapes stable across steps
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(
                hparams['batch_size'], drop_remainder=True
            )
            eval_result = model.evaluate(val_ds, verbose=0)
            val_loss = None
            if eval_result is None or len(eval_result) == 0:
                print(f'Warning: eval_result is None for fold {fold_idx+1}')
                val_loss = 1e6
            else:
                val_loss = float(eval_result if np.isscalar(eval_result) else eval_result[0])

            val_losses.append(val_loss)
            print(f'Fold {fold_idx+1}/{self.n_splits} val_loss: {val_loss}')

            # Do not clear session here to avoid invalidating TF function caches each fold
            del model

        # Clear session once per trial to free memory without excessive retracing
        keras.backend.clear_session()

        mean_loss = np.mean(val_losses)
        print(f'Mean val_loss for this trial: {mean_loss}')
        print(f'Time taken: {time.time() - start_time:.2f} seconds = +-{round((time.time() - start_time) / 60)} minutes')
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