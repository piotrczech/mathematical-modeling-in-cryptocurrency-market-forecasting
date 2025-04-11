import numpy as np
from hyperopt import STATUS_OK, fmin, tpe, Trials, space_eval

class HyperparameterOptimizationService:
    def __init__(self, X_train, y_train, X_val, y_val, build_model: callable):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.build_model = build_model

    def objective(self, hparams):
        model = self.build_model(hparams)

        model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=hparams['epochs'],
            batch_size=hparams['batch_size'],
        )

        # Ocena modelu
        val_loss = model.evaluate(self.X_val, self.y_val, verbose=0)

        return {
            'status': STATUS_OK,
            'loss': val_loss,
            'model': model,
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

        model = trials.best_trial['result']['model']

        return space_eval(space, best), model, trials