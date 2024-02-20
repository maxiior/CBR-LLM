import time
from logs import Logger
import os
import pandas as pd

class Experiment():
    def __init__(self, experiment_configs) -> None:
        self.logger = Logger()
        self.experiment_configs = experiment_configs.experiment

    def run(self, model, dataset, dataset_name):
        start_time = time.time()
        self.logger.save_json(self.experiment_configs.name, dataset_name, model.get_info())

        responses = pd.DataFrame(columns=['target_recipe', 'response'])

        file_name = f'./logs/{self.experiment_configs.name}_results.csv'
        if os.path.isfile(file_name): 
            print('INFO - experiment already exists.')
        else:
            for idx, row in dataset.iterrows():
                try:
                    id = row['id']
                    response = model.send_request(row['input'])
                    
                    responses.loc[len(responses)] = [id, response]

                    if idx % 20 == 0:
                        self.logger.add_results_to_df(self.experiment_configs.name, responses)
                        responses = pd.DataFrame(columns=['target_recipe', 'response'])
                except:
                    print(f'ERR - something went wrong with id: {row["id"]}')

            end_time = time.time()
            
            self.logger.modify_json(self.experiment_configs.name, 'time', end_time-start_time)