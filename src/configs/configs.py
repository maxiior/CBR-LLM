from easydict import EasyDict as edict
import json

class Configs():

    @classmethod
    def get_main_configs(cls):
        with open('configs/configs.json', 'r') as file:
            return edict(json.load(file))
        
    @classmethod
    def get_experiments_configs(cls):
        experiments_configs = []
        configs = cls.get_main_configs()

        for i in configs.experiments:
            with open(f'configs/experiments/{i}', 'r') as file:
                experiments_configs.append(edict(json.load(file)))
        
        return experiments_configs