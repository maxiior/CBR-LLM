
import pandas as pd


def compose(factual, masked) -> str:
    comp = [f"name: {e[2]} ; ingredients: {e[4]} ; preparation:{e[3]}" for e in factual.values[:-1]]
    comp = comp + [f"name: {e[2]} ; ingredients: {e[4]} ; preparation:" for e in [masked.values[-1]]]
    return " ; ".join(comp)


def load_all_datasets(datasets_path:str="datasets/") -> tuple:
    additional_cbr_1 = pd.read_csv(f"{datasets_path}_additional_cbr_1.csv")
    additional_cbr_2 = pd.read_csv(f"{datasets_path}_additional_cbr_2.csv")
    additional_validation = pd.read_csv(f"{datasets_path}_additional_validation.csv")
    cbr_augmentation = pd.read_csv(f"{datasets_path}_cbr_augmentation.csv")
    cbr_database = pd.read_csv(f"{datasets_path}_cbr_database.csv")
    model_training = pd.read_csv(f"{datasets_path}_model_training.csv")
    validation = pd.read_csv(f"{datasets_path}_validation.csv")

    return additional_cbr_1, additional_cbr_2, additional_validation, cbr_augmentation, cbr_database, model_training, validation