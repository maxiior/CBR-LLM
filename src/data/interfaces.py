import pandas as pd
import abc
import random
import os


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

    def _check_if_files_exists(self):
        keys = self.datasets_configs.proportions.keys()
        prefix = self.datasets_configs.target_files_prefix
        number_of_files = 0

        for i in keys:
            if os.path.exists(f'../dataset/{prefix}_{i}.csv'):
                number_of_files += 1

        return number_of_files == len(keys)

    def create_structured_datasets(self):
        if self._check_if_files_exists():
            keys = self.datasets_configs.proportions.keys()
            datasets = {}
            for i in keys:
                datasets[i] = pd.read_csv(f"../dataset/{self.datasets_configs.target_files_prefix}_{i}.csv")
            return datasets
        else:
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
            v.to_csv(f"../dataset/{self.datasets_configs.target_files_prefix}_{k}.csv", index=False)
    
    def prepare_masked_dataset(self, df):
        original_df = df.copy
        for i in range(len(df)):
            df.at[i, random.choice(self.datasets_configs.columns_to_mask)] = self.datasets_configs.mask_tag
        return df, original_df