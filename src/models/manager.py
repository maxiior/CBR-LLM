from .gpt import GPTConnector
from .llama import LlamaCPP

class ModelManager():
    def __init__(self, experiment_configs):
        self.model_configs = experiment_configs.model
    
    def get_model(self):
        model_type = self.model_configs.type
        if model_type == 'gpt':
            return GPTConnector(self.model_configs)
        elif model_type == 'llama':
            return LlamaCPP()
        elif model_type == 'bart':
            pass
        else:
            raise ValueError(f'{model_type} mode does not exists.')
