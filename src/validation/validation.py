from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

class Validation():
    def __init__(self, experiment_configs) -> None:
        self.validation_configs = experiment_configs.validation
        self.prompt_configs = experiment_configs.prompt

    def _bleu(self, original:str, generated:str) -> float:
        return round(sentence_bleu([original.split()], generated.split()), 6)
    
    def _meteor(self, original:str, generated:str) -> float:
        return round(meteor_score([original.split()], generated.split()), 6)

    def _prepare_original(self, original):
        response_structure = self.prompt_configs.response_structure
        for i in self.prompt_configs.tags:
            if i.source == 'dataset':
                response_structure = response_structure.replace(i.value, original[i.column])
        return response_structure

    def validate(self, original, generated:str):
        original = self._prepare_original(original)
        
        self._bleu(original, generated) >= self.validation_configs.bleu_threshold
        self._meteor(original, generated) >= self.validation_configs.meteor_threshold





