import pandas as pd
import abc
import random

class Dataset(abc.ABC):
    def __init__(self, experiment_configs):
        self.datasets_configs = experiment_configs.datasets

    @abc.abstractmethod
    def _preprocessing_dataframe(self, df):
        raise NotImplementedError()

    def _load_dataset(self):
        df = pd.read_csv(f"../dataset/{self.datasets_configs.file_name}")
        return df[self.datasets_configs.columns_to_extract]

    def _shuffle_dataframe(self, df):
        return df.sample(frac=1).reset_index(drop=True)

    def create_structured_datasets(self):
        dataset = self._load_dataset()
        dataset = self._preprocessing_dataframe(dataset)
        dataset = self._shuffle_dataframe(dataset)

        current_row, datasets = 0, {}

        for k, v in self.datasets_configs.proportions.items():
            rows = int(v * len(dataset))
            datasets[k] = dataset.iloc[current_row:current_row+rows]
            current_row += rows

        self._save_datasets(datasets)
        return datasets
    
    def _save_datasets(self, datasets):
        for k, v in datasets.items():
            v.to_csv(f"../dataset/{k}.csv", index=False)
    
    def prepare_masked_dataset(self, df):
        original_df = df.copy
        columns = df.columns
        for _, row in df.iterrows():
            row[random.choice(columns)] = self.datasets_configs.mask_tag
        return df, original_df