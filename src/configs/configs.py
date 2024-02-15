from easydict import EasyDict as edict
import json

class Configs():

    @classmethod
    def get(cls):
        with open('configs.json', 'r') as file:
            return edict(json.load(file))