import pandas as pd
import os
import json


class Logger():
    def __init__(self) -> None:
        pass

    def add_results_to_df(self, prefix, df):
        file_name = f'./logs/{prefix}_results.csv'
        if not os.path.isfile(file_name):
            pd.DataFrame(columns=['target_recipe', 'response']).to_csv(file_name, index=False)
        df.to_csv(file_name, mode='a', index=False, header=False)
    
    def save_json(self, prefix, dataset_name, model_info):
        d = {
            "dataset_name": dataset_name,
            "model": model_info['model'],
            "model_path": model_info['model_path'],
            "n_gpu_layers": model_info['n_gpu_layers'],
            "n_batch": model_info['n_batch'],
            "n_ctx": model_info['n_ctx']
        }

        with open(f"./logs/{prefix}_sample.json", "w") as outfile: 
            json.dump(d, outfile, indent=4)
    
    def modify_json(self, prefix, key, value):
        with open(f"./logs/{prefix}_sample.json", 'r') as file:
            data = json.load(file)
        data[key] = value
        with open(f"./logs/{prefix}_sample.json", 'w') as file:
            json.dump(data, file, indent=4)

        