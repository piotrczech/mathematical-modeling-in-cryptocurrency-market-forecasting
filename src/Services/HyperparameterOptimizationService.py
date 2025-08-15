from sklearn.model_selection import TimeSeriesSplit
from hyperopt import STATUS_OK, fmin, tpe, Trials, space_eval
import json
import time
import keras.backend
import numpy as np
from tensorflow.keras import losses as keras_losses
from rich.table import Table
from tensorflow import keras
import tensorflow as tf
from src.Preprocessing import RobustFeatureScaler, TargetScaler
from src.Services.MetricsService import MetricsService
from src.Services.ConsoleService import display_table, display_metrics_table

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
        hparams['loss_method'] = self.loss_function.__name__
        val_losses = []
        all_original_metrics = {'MSE': [], 'MAE': [], 'RMSE': [], 'MAPE': [], 'HUBER': []}

        print(f'\nEvaluating hyperparameters: {json.dumps(hparams, indent=2)}\n')
        start_time = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(tss.split(self.X)):
            X_tr, X_val = self.X[train_idx], self.X[val_idx]
            y_tr, y_val = self.y[train_idx], self.y[val_idx]

            # Assert shapes
            assert len(X_tr.shape) == 3, f'X_tr must be 3D, got {X_tr.shape}'
            assert len(y_tr.shape) == 2, f'y_tr must be 2D, got {y_tr.shape}'

            x_scaler = RobustFeatureScaler().fit(X_tr)
            X_tr = x_scaler.transform(X_tr)
            X_val = x_scaler.transform(X_val)

            y_scaler = TargetScaler().fit(y_tr)
            y_tr = y_scaler.transform(y_tr)
            y_val = y_scaler.transform(y_val)

            model = self.build_model(hparams)

            # Train
            callbacks = [
                keras.callbacks.EarlyStopping(patience=round(hparams['epochs']/4), restore_best_weights=True, monitor='val_loss'),
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

            y_pred = model.predict(X_val, batch_size=hparams['batch_size'], verbose=0)
            original_scale_metrics = MetricsService.calculate_metrics(y_val, y_pred, y_scaler)
            for metric_name in all_original_metrics:
                all_original_metrics[metric_name].append(original_scale_metrics[metric_name])
            
            val_losses.append(val_loss)
            
            print(f"Fold {fold_idx+1}/{self.n_splits} - Val Loss (scaled): {val_loss:.6f}")

            del model

        # Clear session once per trial to free memory without excessive retracing
        keras.backend.clear_session()

        mean_loss = np.mean(val_losses)
        
        mean_original_metrics = {
            metric_name: np.mean(values) for metric_name, values in all_original_metrics.items()
        }
        
        print(f"Summary of trial")
        print(f"Time taken: {time.time() - start_time:.2f}s (~{round((time.time() - start_time) / 60)}min)")
        display_metrics_table(
            mean_original_metrics,
            title="Average metrics (original scale)"
        )
        print() 

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
        
        # Wyświetl finalne podsumowanie optymalizacji w ładnej tabeli
        final_summary_rows = [
            ["Best Trial", f"#{best_trial_idx + 1}"],
            ["Best Mean Loss (scaled)", f"{best_trial['result']['loss']:.6f}"],
            ["Total Evaluations", f"{len(trials.trials)}"]
        ]
        
        display_table(
            title="Final results",
            columns=["Parameter", "Value"],
            rows=final_summary_rows
        )

        return space_eval(space, best), trials