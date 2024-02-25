from .gpt import GPTConnector
from .llama import LlamaCPP

class ModelManager():
    def __init__(self, experiment_configs):
        self.model_configs = experiment_configs.model
    
    def get_model(self):
        model_name = self.model_configs.name
        if model_name == 'gpt':
            return GPTConnector(self.model_configs)
        elif model_name == 'llama':
            return LlamaCPP(self.model_configs)
        elif model_name == 'bart':
            pass
        else:
            raise ValueError(f'{model_name} mode does not exists.')
