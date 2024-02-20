import pandas as pd
from logs import Logger
import time
import os

class LlamaCPPMock():
    def __init__(self, model_path="models/llama-13b-hf_q8_0.gguf", n_gpu_layers=41, n_batch=1024, n_ctx=2048) -> None:
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.model = None

    def send_request(self, X, break_word:str = " name:") -> str:
        return X

    def get_info(self) -> str:
        return {
            "model":"LLama-13b-hf-q8_0",
            "model_path":self.model_path,
            "n_gpu_layers":self.n_gpu_layers,
            "n_batch":self.n_batch,
            "n_ctx":self.n_ctx,
        }

log = Logger()

def read_dataset(name):
    return pd.read_csv(f"./{name}")

def run(model, dataset_name, experiment_name):
    start_time = time.time()
    
    log.save_json(experiment_name, dataset_name, model.get_info())
    dataset = read_dataset(dataset_name)

    responses = pd.DataFrame(columns=['response', 'target_recipe'])

    file_name = f'./logs/{experiment_name}_results.csv'
    if os.path.isfile(file_name): raise Exception('experiment already exists.')

    for idx, row in dataset.iterrows():
        # try:
        id = row['target_recipe']
        response = model.send_request(row['input'])
        
        responses.loc[len(responses)] = [response, id]

        if idx % 20 == 0:
            log.add_results_to_df(experiment_name, responses)
            responses = pd.DataFrame(columns=['response', 'target_recipe'])
        # except Exception as e:
        #     print(e)
        #     pass

    log.add_results_to_df(experiment_name, responses)
    end_time = time.time()
    
    log.modify_json(experiment_name, 'time', end_time-start_time)




