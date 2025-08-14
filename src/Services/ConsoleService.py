import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm


ACTION_BACK = '0'

# Main menu
ACTION_SEARCH_TRAIN = 'a'
ACTION_SEARCH_SAVE = 'b'
ACTION_LOAD_TRAIN = 'c'
ACTION_LOAD_PREDICT = 'd'

# Model menu
ACTION_CHOICE_MODEL_MLP = '1'
ACTION_CHOICE_MODEL_LSTM = '2'

console = Console()

def clear_console():
    console.clear()

def show_welcome():
    welcome_text = Text(r"""
    _______  _______  ______   _______  _        _______  _       _________
    (       )(  ___  )(  __  \ (  ____ \( \      (  ____ \( \      \__   __/
    | () () || (   ) || (  \  )| (    \/| (      | (    \/| (         ) (   
    | || || || |   | || |   ) || (__    | |      | |      | |         | |   
    | |(_)| || |   | || |   | ||  __)   | |      | |      | |         | |   
    | |   | || |   | || |   ) || (      | |      | |      | |         | |   
    | )   ( || (___) || (__/  )| (____/\| (____/\| (____/\| (____/\___) (___
    |/     \|(_______)(______/ (_______/(_______/(_______/(_______/\_______/
    """, style="bold blue")
    
    panel = Panel(welcome_text, title="Model CLI - Train and Optimize Crypto Models", border_style="blue")
    console.print(panel)

def show_model_choice_menu():
    menu_text = f"""
{ACTION_CHOICE_MODEL_MLP}. MLP (Multi-Layer Perceptron)
{ACTION_CHOICE_MODEL_LSTM}. LSTM (Long Short-Term Memory)
{ACTION_BACK}. Back
    """
    panel = Panel(menu_text, title="Select Model Structure", border_style="green")
    console.print(panel)

def get_model_choice() -> str:
    while True:
        choice = Prompt.ask("[bold green]Choice (0-2)[/bold green]", choices=[ACTION_CHOICE_MODEL_MLP, ACTION_CHOICE_MODEL_LSTM, ACTION_BACK], show_choices=False)
        if choice == ACTION_BACK:
            return "exit"
        elif choice == ACTION_CHOICE_MODEL_MLP:
            return "mlp"
        elif choice == ACTION_CHOICE_MODEL_LSTM:
            return "lstm"
        console.print("[red]Invalid choice.[/red]")

def show_action_menu():
    menu_text = f"""
{ACTION_SEARCH_TRAIN}. Search for model from scratch (Phase 1) and execute full code
{ACTION_SEARCH_SAVE}. Search for model from scratch (Phase 1) and save best model
{ACTION_LOAD_TRAIN}. Load model and train
{ACTION_LOAD_PREDICT}. Load model and predict
{ACTION_BACK}. Exit
    """
    panel = Panel(menu_text, title="Select Action", border_style="green")
    console.print(panel)

def get_action_choice() -> str:
    while True:
        choice = Prompt.ask("[bold green]Choice (a-d or 0)[/bold green]", choices=[ACTION_SEARCH_TRAIN, ACTION_SEARCH_SAVE, ACTION_LOAD_TRAIN, ACTION_LOAD_PREDICT, ACTION_BACK], show_choices=False).lower()
        if choice in [ACTION_SEARCH_TRAIN, ACTION_SEARCH_SAVE, ACTION_LOAD_TRAIN, ACTION_LOAD_PREDICT, ACTION_BACK]:
            return choice
        console.print("[red]Invalid choice.[/red]")

def prompt_for_model_name(default: str = None) -> str:
    models_dir = Path("assets/models")
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)

    models_list = [f.stem for f in models_dir.glob("*.keras")]

    table = Table(title="Models")
    table.add_column("#", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Date", style="yellow")
    for model in models_list:
        model_path = models_dir / f"{model}.keras"
        mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_path.stat().st_mtime))
        table.add_row(str(models_list.index(model) + 1), model, mod_time)
    panel = Panel(table, title="Available models")
    console.print(panel)

    if not models_list:
        return Prompt.ask("[bold green]Enter model name[/bold green]", default=default or time.strftime("%Y_%m_%d-%H_%M_%S"))
    
    choices = [str(i) for i in range(1, len(models_list) + 1)]
    choice = Prompt.ask(
        "\n[bold green]Choose model number or enter new name[/bold green]",
        choices=choices,
        show_choices=True,
        default=""
    )
    
    if choice in choices:
        return models_list[int(choice) - 1]
    return choice or (default or time.strftime("%Y_%m_%d-%H_%M_%S"))
    
def confirm_action(message: str) -> bool:
    return Confirm.ask(f"[yellow]{message}[/yellow]")

def display_message(message: str, style: str = "green"):
    console.print(Panel(message, border_style=style))

def display_table(title: str, columns: list, rows: list):
    table = Table(title=title)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*row)
    console.print(table) 