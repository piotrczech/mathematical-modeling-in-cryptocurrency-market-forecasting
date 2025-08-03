import os
import time
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from src.Services.DataProcessService import DataProcessService
from src.Services.LoadDataFrameService import LoadDataFrameService
from src.Services.HyperparameterOptimizationService import HyperparameterOptimizationService
from src.Visualizations.LearningCurveVisualizer import LearningCurveVisualizer
from src.Visualizations.PredictionRangeAccuracyVisualizer import PredictionVisualizationService
from src.Visualizations.PredictionRangeEvaluatorVisualizer import PredictionRangeEvaluatorVisualizer
from src.Services.ConsoleService import *
from rich.table import Table
from src.Models import MLPModel, LSTMModel, ResMLPModel
from tensorflow.keras import losses as keras_losses

# Default configurations
SEQUENCE_LENGTH = 30
TEST_DAYS = 90
DATA_PATH = 'assets/crypto-data.csv'
MODELS_DIR = os.path.join("assets", "models")

os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    display_message("Loading data...")
    df = LoadDataFrameService(DATA_PATH).load_dataframe()
    data_processor = DataProcessService(sequence_length=SEQUENCE_LENGTH, test_days=TEST_DAYS)
    X_full, y_full, full_dates = data_processor.create_sequences(df)
    X_trainval, y_trainval, dates_trainval, X_test, y_test, dates_test = data_processor.split_data(X_full, y_full, full_dates)
    display_message("Data loaded")
    return X_trainval, y_trainval, dates_trainval, X_test, y_test, dates_test

def scale_data(X_train, y_train, X_val=None, y_val=None, no_val=False):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scale_X(X_train, scaler_X, fit=True)
    y_train_scaled = scale_y(y_train, scaler_y, fit=True)

    if no_val:
        return X_train_scaled, y_train_scaled, scaler_X, scaler_y

    X_val_scaled = scale_X(X_val, scaler_X, fit=False)
    y_val_scaled = scale_y(y_val, scaler_y, fit=False)
    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scaler_X, scaler_y

def scale_X(X, scaler_X, fit=False):
    assert len(X.shape) == 3
    orig_shape = X.shape
    X_resh = X.reshape(-1, orig_shape[-1])
    if fit:
        X_scaled = scaler_X.fit_transform(X_resh)
    else:
        X_scaled = scaler_X.transform(X_resh)
    return X_scaled.reshape(orig_shape)

def scale_y(y, scaler_y, fit=False):
    assert len(y.shape) == 2
    if fit:
        y_scaled = scaler_y.fit_transform(y)
    else:
        y_scaled = scaler_y.transform(y)
    return y_scaled

def display_parameters_table(params: dict):
    table = Table(title="Best Parameters Found", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    for param, value in params.items():
        table.add_row(str(param), str(value))
    
    console.print(table)

def perform_hyperopt(X_trainval, y_trainval, build_model, space, loss_method, max_evals=25, folds=5):
    display_message("Searching for best architecture...")
    loss_fn = keras_losses.get(loss_method)  # Pobiera callable na podstawie stringa, np. 'huber' -> Huber()
    hopt_service = HyperparameterOptimizationService(X_trainval, y_trainval, build_model, loss_fn, n_splits=folds)
    best_params, trials = hopt_service.optimize(space, max_evals=max_evals)
    display_parameters_table(best_params)
    return best_params

def get_model_save_path(action: str) -> str:
    default_name = time.strftime("%Y_%m_%d-%H_%M_%S")
    if action == ACTION_SEARCH_SAVE:
        default_name = f"untrained_{default_name}"
    else:
        default_name = f"trained_{default_name}"
    
    model_name = Prompt.ask(
        "[bold green]Enter model name[/bold green] (or press Enter for default)",
        default=default_name
    )
    return f"{MODELS_DIR}/{model_name}.keras"

def train_model(model, X_train_scaled, y_train_scaled, X_val_scaled=None, y_val_scaled=None, checkpoint_path=None, epochs=None):
    display_message("Final training of the model...")
    callbacks = [
        # keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss'),
        # keras.callbacks.EarlyStopping(patience=500, restore_best_weights=True, monitor='val_loss'),
    ]
    start_time = time.time()
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled) if X_val_scaled is not None else None,
        epochs=epochs,
        batch_size=model.get_config().get('batch_size', 32),  # Assume from params
        callbacks=callbacks,
        verbose=1
    )
    end_time = time.time()
    
    try:
        model.save(checkpoint_path)
        display_message(f'Best model saved to {checkpoint_path}')
    except Exception as e:
        display_message(f'Error saving model to {checkpoint_path}: {str(e)}', style="red")
    
    return history, end_time - start_time

def evaluate_model(model, X_test, y_test, scaler_X, scaler_y, dates_test=None, is_prediction_only=False):
    message = "Making predictions" if is_prediction_only else "Evaluation on test data"
    display_message(message)
    X_test_scaled = scale_X(X_test, scaler_X, fit=False)

    predictions_scaled = model.predict(X_test_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test_inv = y_test
    
    visualizer = PredictionVisualizationService(y_test_inv, predictions)
    visualizer.visualize()
    evaluator = PredictionRangeEvaluatorVisualizer(y_test_inv, predictions)
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

    console.print(table)

    try:
        LearningCurveVisualizer(history).visualize()
    except Exception as e:
        display_message(f"Failed to generate learning curve: {e}", style="red")

def main():
    while True:
        clear_console()
        show_welcome()
        show_action_menu()
        action = get_action_choice()
        
        if action == ACTION_BACK:
            break
            
        clear_console()
            
        if action in [ACTION_SEARCH_TRAIN, ACTION_SEARCH_SAVE]:
            show_model_choice_menu()
            model_type = get_model_choice()
            if model_type == "exit":
                continue
                
            # After model choice, prompt for config
            display_message("Configure training parameters (press Enter to use defaults)")
            hyperopt_epochs = int(Prompt.ask(
                "[bold green]HYPEROPT_EPOCHS[/bold green] (Number of epochs for hyperparameter optimization)",
                default=str(30)
            ))
            final_epochs = int(Prompt.ask(
                "[bold green]FINAL_EPOCHS[/bold green] (Number of epochs for final model training)",
                default=str(100)
            ))
            loss_method = Prompt.ask(
                "[bold green]LOSS_METHOD[/bold green] (Loss function to use, e.g., 'huber', 'mse')",
                default="huber"
            )
            folds = int(Prompt.ask(
                "[bold green]FOLDS[/bold green] (Number of folds for cross-validation in hyperopt)",
                default=str(5)
            ))
            max_evals = int(Prompt.ask(
                "[bold green]MAX_EVALS[/bold green] (Maximum evaluations for hyperparameter search)",
                default=str(25)
            ))
            
            if model_type == "mlp":
                model = MLPModel(sequence_length=SEQUENCE_LENGTH)
                space = model.get_space(epochs=hyperopt_epochs, loss_method=loss_method)
            elif model_type == "lstm":
                model = LSTMModel(sequence_length=SEQUENCE_LENGTH)
                space = model.get_space(epochs=hyperopt_epochs, loss_method=loss_method)
            elif model_type == "res_mlp":
                model = ResMLPModel(sequence_length=SEQUENCE_LENGTH)
                space = model.get_space(epochs=hyperopt_epochs, loss_method=loss_method)
        
        clear_console()
        
        X_trainval, y_trainval, _, X_test, y_test, dates_test = load_data()
        
        if action in [ACTION_SEARCH_TRAIN, ACTION_SEARCH_SAVE]:
            best_params = perform_hyperopt(X_trainval, y_trainval, model.build_model, space, loss_method=loss_method, max_evals=max_evals, folds=folds)
            model = model.build_model(best_params)
            
            if action == ACTION_SEARCH_SAVE:
                checkpoint_path = get_model_save_path(action)
                model.save(checkpoint_path)
                display_message(f"Best model saved without training to {checkpoint_path}")
                Prompt.ask("Press enter to return to main menu")
                continue
        
        elif action in [ACTION_LOAD_TRAIN, ACTION_LOAD_PREDICT]:
            model_name = prompt_for_model_name()
            checkpoint_path = f"{MODELS_DIR}/{model_name}.keras"
            if not os.path.exists(checkpoint_path):
                display_message("Model file not found.", style="red")
                Prompt.ask("Press enter to return to main menu")
                continue
            model = keras.models.load_model(checkpoint_path)
        
        if action == ACTION_LOAD_PREDICT:
            # Only predict
            X_tr_scaled, y_tr_scaled, scaler_X, scaler_y = scale_data(X_trainval, y_trainval, no_val=True)
            evaluate_model(model, X_test, y_test, scaler_X, scaler_y, dates_test, is_prediction_only=True)
        else:
            # Train
            checkpoint_path = get_model_save_path(action)
            X_tr_scaled, y_tr_scaled, scaler_X, scaler_y = scale_data(X_trainval, y_trainval, no_val=True)
            history, training_time = train_model(model, X_tr_scaled, y_tr_scaled, checkpoint_path=checkpoint_path, epochs=final_epochs)
            summarize_training_history(history, training_time)
            
            # Evaluate
            evaluate_model(model, X_test, y_test, scaler_X, scaler_y, dates_test)
        
        Prompt.ask("Press enter to return to main menu")

if __name__ == "__main__":
    main()
