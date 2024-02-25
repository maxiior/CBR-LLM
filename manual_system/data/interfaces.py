import pandas as pd
import abc
import random
import os
from copy import deepcopy


class Dataset(abc.ABC):
    def __init__(self, source_file_name, save_files_prefix, columns_to_extract, proportions):
        self.source_file_name = source_file_name
        self.save_files_prefix = save_files_prefix
        self.columns_to_extract = columns_to_extract
        self.proportions = proportions

    @abc.abstractmethod
    def _preprocessing_dataframe(self, df):
        raise NotImplementedError()

    def _load_dataset(self):
        df = pd.read_csv(f"../dataset/{self.source_file_name}")
        return df[self.columns_to_extract]

    def _shuffle_dataframe(self, df):
        return df.sample(frac=1).reset_index(drop=True)

    def _check_if_files_exists(self):
        keys = self.proportions.keys()
        prefix = self.save_files_prefix
        number_of_files = 0

        for i in keys:
            if os.path.exists(f'data/datasets/{prefix}_{i}.csv'):
                number_of_files += 1

        return number_of_files == len(keys)

    def save_data(self, dataset, dataset_name):
        dataset.to_csv(f"data/datasets/{self.save_files_prefix}_{dataset_name}.csv", mode='a', index=False, header=False)
        
    def create_structured_datasets(self):
        if self._check_if_files_exists():
            print('INFO - datasets for this experiment already exists.')
            keys = self.proportions.keys()
            datasets = {}
            for i in keys:
                datasets[i] = pd.read_csv(f"data/datasets/{self.save_files_prefix}_{i}.csv")
            return datasets
        else:
            dataset = self._load_dataset()
            dataset = self._preprocessing_dataframe(dataset)
            dataset = self._shuffle_dataframe(dataset)

            current_row, datasets = 0, {}

            for k, v in self.proportions.items():
                rows = int(v * len(dataset))
                datasets[k] = dataset.iloc[current_row:current_row+rows]
                current_row += rows

            self._save_datasets(datasets)
            return datasets
    
    def _save_datasets(self, datasets):
        for k, v in datasets.items():
            v.to_csv(f"data/datasets/{self.save_files_prefix}_{k}.csv", index=False)

    def prepare_cbr_dataset(self, dataset, columns):
        ids = dataset[['id']]['id'].to_list()
        dataset = dataset[columns]

        columns_data = [dataset[i].to_list() for i in columns]
        zipped_data = [list(zipped) for zipped in zip(*columns_data)]
        

        return [" ; ".join([f'{c}: {j}' for c, j in zip(columns, i)]) for i in zipped_data], ids