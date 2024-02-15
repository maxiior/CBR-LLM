import pandas as pd

class DataCreator():
    def __init__(self, configs) -> None:
        self.configs = configs

    def load_dataset(self, path):
        df = pd.read_csv("../../RAW_recipes.csv")
        df = df.dropna()

    def create(self):
        dataset = self.load_dataset()

