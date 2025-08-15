import os
import time
import numpy as np
from tensorflow import keras
from src.Preprocessing import RobustFeatureScaler, TargetScaler
from src.Services import DataProcessService, LoadDataFrameService, HyperparameterOptimizationService, ConsoleService
from src.Visualizations.LearningCurveVisualizer import LearningCurveVisualizer
from src.Visualizations.PredictionRangeAccuracyVisualizer import PredictionVisualizationService
from src.Visualizations.PredictionRangeEvaluatorVisualizer import PredictionRangeEvaluatorVisualizer
from rich.table import Table
from src.Models import MLPModel, LSTMModel, MLPStochasticModel, LSTMStochasticModel
from src.Models.base_stochastic_model import StochasticModule
from tensorflow.keras import losses as keras_losses

# Default configurations
SEQUENCE_LENGTH = 30
TEST_DAYS = 90
DATA_PATH = 'assets/crypto-data.csv'
MODELS_DIR = os.path.join("assets", "models")

os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    ConsoleService.display_message("Loading data...")
    df = LoadDataFrameService(DATA_PATH).load_dataframe()
    data_processor = DataProcessService(sequence_length=SEQUENCE_LENGTH, test_days=TEST_DAYS)
    X_full, y_full, full_dates = data_processor.create_sequences(df)
    X_trainval, y_trainval, X_test, y_test = data_processor.split_data(X_full, y_full)
    ConsoleService.display_message("Data loaded")
    return X_trainval, y_trainval, X_test, y_test

def scale_for_training(X_trainval, y_trainval, X_test, y_test):
    x_scaler = RobustFeatureScaler().fit(X_trainval)
    y_scaler = TargetScaler().fit(y_trainval)
    X_tr_s = x_scaler.transform(X_trainval)
    X_te_s = x_scaler.transform(X_test)
    y_tr_s = y_scaler.transform(y_trainval)
    y_te_s = y_scaler.transform(y_test)
    return X_tr_s, y_tr_s, X_te_s, y_te_s, x_scaler, y_scaler

def display_parameters_table(params: dict):
    table = Table(title="Best Parameters Found", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    for param, value in params.items():
        table.add_row(str(param), str(value))
    
    ConsoleService.console.print(table)

def perform_hyperopt(X_trainval, y_trainval, build_model, space, loss_method, max_evals=25, folds=5):
    ConsoleService.display_message("Searching for best architecture...")
    loss_fn = keras_losses.get(loss_method)  # Pobiera callable na podstawie stringa, np. 'huber' -> Huber()
    hopt_service = HyperparameterOptimizationService(X_trainval, y_trainval, build_model, loss_fn, n_splits=folds)
    best_params, trials = hopt_service.optimize(space, max_evals=max_evals)
    display_parameters_table(best_params)
    return best_params

def get_model_save_path(action: str) -> str:
    default_name = time.strftime("%Y_%m_%d-%H_%M_%S")
    if action == ConsoleService.ACTION_SEARCH_SAVE:
        default_name = f"untrained_{default_name}"
    else:
        default_name = f"trained_{default_name}"
    
    model_name = ConsoleService.Prompt.ask(
        "[bold green]Enter model name[/bold green] (or press Enter for default)",
        default=default_name
    )
    return f"{MODELS_DIR}/{model_name}.keras"

def train_model(model, X_train, y_train, X_val=None, y_val=None, checkpoint_path=None, epochs=None):
    ConsoleService.display_message("Final training of the model...")
    callbacks = []

    if ConsoleService.Prompt.ask("Do you want to use early stopping?", choices=["y", "n"], default="y") == "y":
        patience = ConsoleService.IntPrompt.ask("Enter patience for early stopping", default=50)
        callbacks.append(keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True, monitor='val_loss', verbose=1))
    
    if ConsoleService.Prompt.ask("Do you want to use model checkpointing?", choices=["y", "n"], default="n") == "y":
        callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss'))

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=epochs,
        batch_size=model.get_config().get('batch_size', 32),  # Assume from params
        callbacks=callbacks,
        verbose=1
    )
    end_time = time.time()
    
    try:
        model.save(checkpoint_path)
        ConsoleService.display_message(f'Best model saved to {checkpoint_path}')
    except Exception as e:
        ConsoleService.display_message(f'Error saving model to {checkpoint_path}: {str(e)}', style="red")
    
    return history, end_time - start_time

def evaluate_model(model, X_test, y_test, y_scaler=None, is_prediction_only=False):
    message = "Making predictions" if is_prediction_only else "Evaluation on test data"
    ConsoleService.display_message(message)

    # Use stable batch size to avoid retracing due to last partial batch
    predictions = model.predict(X_test, batch_size=model.get_config().get('batch_size', 32))

    if y_scaler is not None:
        try:
            predictions_inv = y_scaler.inverse_transform(predictions)
            y_test_inv = y_scaler.inverse_transform(y_test)
        except Exception:
            predictions_inv = predictions
            y_test_inv = y_test
    else:
        predictions_inv = predictions
        y_test_inv = y_test

    visualizer = PredictionVisualizationService(y_test_inv, predictions_inv)
    visualizer.visualize()
    evaluator = PredictionRangeEvaluatorVisualizer(y_test_inv, predictions_inv)
    evaluator.print_summary()
    evaluator.plot_range_accuracy()

def summarize_training_history(history, training_time=None):
    train_losses = history.history.get("loss", [])
    val_losses = history.history.get("val_loss", [])
    
    table = Table(title="Training Summary", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Value", style="green", justify="left")

    if val_losses:  # Only show validation metrics if we have validation data
        best_epoch = int(np.argmin(val_losses)) + 1
        best_val_loss = np.min(val_losses)
        table.add_row("Best Epoch", str(best_epoch))
        table.add_row("Min Val Loss", f"{best_val_loss:.5f}")
        table.add_row("Final Val Loss", f"{val_losses[-1]:.5f}")
    
    if training_time is not None:
        table.add_row("Training Time", f"{training_time:.2f} s")
    
    table.add_row("Final Train Loss", f"{train_losses[-1]:.5f}")

    ConsoleService.console.print(table)

    try:
        LearningCurveVisualizer(history).visualize()
    except Exception as e:
        ConsoleService.display_message(f"Failed to generate learning curve: {e}", style="red")

def main():
    while True:
        ConsoleService.clear_console()
        ConsoleService.show_welcome()
        ConsoleService.show_action_menu()
        action = ConsoleService.get_action_choice()
        
        if action == ConsoleService.ACTION_BACK:
            break
            
        ConsoleService.clear_console()
            
        if action in [ConsoleService.ACTION_SEARCH_TRAIN, ConsoleService.ACTION_SEARCH_SAVE]:
            ConsoleService.show_model_choice_menu()
            model_type = ConsoleService.get_model_choice()
            if model_type == "exit":
                continue
                
            # After model choice, prompt for config
            ConsoleService.display_message("Configure training parameters (press Enter to use defaults)")
            hyperopt_epochs = int(ConsoleService.Prompt.ask(
                "[bold green]HYPEROPT_EPOCHS[/bold green] (Number of epochs for hyperparameter optimization)",
                default=str(30)
            ))
            final_epochs = int(ConsoleService.Prompt.ask(
                "[bold green]FINAL_EPOCHS[/bold green] (Number of epochs for final model training)",
                default=str(100)
            ))
            loss_method = ConsoleService.Prompt.ask(
                "[bold green]LOSS_METHOD[/bold green] (Loss function to use, e.g., 'huber', 'mse')",
                default="huber"
            )
            folds = int(ConsoleService.Prompt.ask(
                "[bold green]FOLDS[/bold green] (Number of folds for cross-validation in hyperopt)",
                default=str(5)
            ))
            max_evals = int(ConsoleService.Prompt.ask(
                "[bold green]MAX_EVALS[/bold green] (Maximum evaluations for hyperparameter search)",
                default=str(25)
            ))
            
            if model_type == "mlp":
                model = MLPModel(sequence_length=SEQUENCE_LENGTH)
                space = model.get_space(epochs=hyperopt_epochs, loss_method=loss_method)
            elif model_type == "lstm":
                model = LSTMModel(sequence_length=SEQUENCE_LENGTH)
                space = model.get_space(epochs=hyperopt_epochs, loss_method=loss_method)
            elif model_type == "mlp_stochastic":
                model = MLPStochasticModel(sequence_length=SEQUENCE_LENGTH)
                space = model.get_space(epochs=hyperopt_epochs, loss_method=loss_method)
            elif model_type == "lstm_stochastic":
                model = LSTMStochasticModel(sequence_length=SEQUENCE_LENGTH)
                space = model.get_space(epochs=hyperopt_epochs, loss_method=loss_method)
        
        ConsoleService.clear_console()
        
        X_trainval, y_trainval, X_test, y_test = load_data()
        
        if action in [ConsoleService.ACTION_SEARCH_TRAIN, ConsoleService.ACTION_SEARCH_SAVE]:
            best_params = perform_hyperopt(X_trainval, y_trainval, model.build_model, space, loss_method=loss_method, max_evals=max_evals, folds=folds)
            model = model.build_model(best_params)
            
            if action == ConsoleService.ACTION_SEARCH_SAVE:
                checkpoint_path = get_model_save_path(action)
                model.save(checkpoint_path)
                ConsoleService.display_message(f"Best model saved without training to {checkpoint_path}")
                ConsoleService.Prompt.ask("Press enter to return to main menu")
                continue
        
        elif action in [ConsoleService.ACTION_LOAD_TRAIN, ConsoleService.ACTION_LOAD_PREDICT]:
            model_name = ConsoleService.prompt_for_model_name()
            checkpoint_path = f"{MODELS_DIR}/{model_name}.keras"
            if not os.path.exists(checkpoint_path):
                ConsoleService.display_message("Model file not found.", style="red")
                ConsoleService.Prompt.ask("Press enter to return to main menu")
                continue
            model = keras.models.load_model(checkpoint_path, custom_objects={"StochasticModule": StochasticModule})
        
        if action == ConsoleService.ACTION_LOAD_PREDICT:
            # Only predict – dopasuj skalery na trainval i użyj do skalowania testu oraz odskalowania predykcji
            X_tr_s, y_tr_s, X_te_s, y_te_s, _, y_scaler = scale_for_training(X_trainval, y_trainval, X_test, y_test)

            X_pred = X_te_s
            y_pred = y_te_s
            
            if ConsoleService.Prompt.ask("Do you want to add days of training data for prediction?", choices=["y", "n"], default="y") == "y":
                days = ConsoleService.IntPrompt.ask("Enter number of days to add", default=90)
                X_pred = np.concatenate([X_tr_s[-days:], X_te_s], axis=0)
                y_pred = np.concatenate([y_tr_s[-days:], y_te_s], axis=0)

            evaluate_model(model, X_pred, y_pred, y_scaler=y_scaler, is_prediction_only=True)
        elif action == ConsoleService.ACTION_LOAD_TRAIN:
            # Train
            train_epochs = int(ConsoleService.Prompt.ask(
                "[bold green]EPOCHS[/bold green] (Number of epochs for model training)",
                default=str(100)
            ))

            checkpoint_path = get_model_save_path(action)
            X_tr_s, y_tr_s, X_te_s, y_te_s, _, y_scaler = scale_for_training(X_trainval, y_trainval, X_test, y_test)
            history, training_time = train_model(model, X_tr_s, y_tr_s, checkpoint_path=checkpoint_path, epochs=train_epochs)
            summarize_training_history(history, training_time)
            
            # Evaluate
            X_pred = X_te_s
            y_pred = y_te_s

            if ConsoleService.Prompt.ask("Do you want to add days of training data for prediction?", choices=["y", "n"], default="y") == "y":
                days = ConsoleService.IntPrompt.ask("Enter number of days to add", default=90)
                X_pred = np.concatenate([X_tr_s[-days:], X_te_s], axis=0)
                y_pred = np.concatenate([y_tr_s[-days:], y_te_s], axis=0)

            evaluate_model(model, X_pred, y_pred, y_scaler=y_scaler)
        else:
            checkpoint_path = get_model_save_path(action)
            X_tr_s, y_tr_s, X_te_s, y_te_s, _, y_scaler = scale_for_training(X_trainval, y_trainval, X_test, y_test)
            history, training_time = train_model(model, X_tr_s, y_tr_s, checkpoint_path=checkpoint_path, epochs=final_epochs)
            summarize_training_history(history, training_time)
            
            # Evaluate
            X_pred = X_te_s
            y_pred = y_te_s

            if ConsoleService.Prompt.ask("Do you want to add days of training data for prediction?", choices=["y", "n"], default="y") == "y":
                days = ConsoleService.IntPrompt.ask("Enter number of days to add", default=90)
                X_pred = np.concatenate([X_tr_s[-days:], X_te_s], axis=0)
                y_pred = np.concatenate([y_tr_s[-days:], y_te_s], axis=0)

            evaluate_model(model, X_pred, y_pred, y_scaler=y_scaler)
        
        ConsoleService.Prompt.ask("Press enter to return to main menu")

if __name__ == "__main__":
    main()
