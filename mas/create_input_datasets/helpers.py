
import pandas as pd



def compose(factual:pd.DataFrame, masked:"pd.Row") -> str:
    comp = [f"name: {e['name']} ; ingredients: {e['ingredients']} ; preparation:{e['steps']}" for i, e in factual.iterrows()]
    comp = comp + [f"name: {masked['name']} ; ingredients: {masked['ingredients']} ; preparation:"]
    return " ; ".join(comp)

def str_rep_no_steps(df_recipes) -> str:
    comp = f"name: {df_recipes['name']} ; ingredients: {df_recipes['ingredients']} ; preparation:"
    return comp

def str_rep(df_recipes) -> str:
    comp = [f"name: {e['name']} ; ingredients: {e['ingredients']} ; preparation:{e['steps']}" for i, e in df_recipes.iterrows()]
    return comp

def load_all_datasets(datasets_path:str="datasets/") -> tuple:
    additional_cbr_1 = pd.read_csv(f"{datasets_path}_additional_cbr_1.csv")
    additional_cbr_2 = pd.read_csv(f"{datasets_path}_additional_cbr_2.csv")
    additional_validation = pd.read_csv(f"{datasets_path}_additional_validation.csv")
    cbr_augmentation = pd.read_csv(f"{datasets_path}_cbr_augmentation.csv")
    cbr_database = pd.read_csv(f"{datasets_path}_cbr_database.csv")
    model_training = pd.read_csv(f"{datasets_path}_model_training.csv")
    validation = pd.read_csv(f"{datasets_path}_validation.csv")

    return additional_cbr_1, additional_cbr_2, additional_validation, cbr_augmentation, cbr_database, model_training, validation