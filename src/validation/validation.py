from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from langchain_community.embeddings import GPT4AllEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


class Validation():
    def __init__(self, experiment_configs) -> None:
        self.validation_configs = experiment_configs.validation
        self.prompt_configs = experiment_configs.prompt

    def _cosine_similarity(self, original:str, generated:str) -> float:
        tokenizer = GPT4AllEmbeddings()
        X = tokenizer.embed_query(generated)
        Y = tokenizer.embed_query(original)
        return round(cosine_similarity([X], [Y]), 6)

    def _bleu(self, original:str, generated:str, version:int=4) -> float:
        weights = [0.25, 0.25, 0.25, 0.25]
        if version == 3:
            weights = [0.33, 0.33, 0.33]
        elif version == 2:
            weights = [0.5, 0.5]

        return round(sentence_bleu([original.split()], generated.split(), weights=weights), 6)
    
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
        
        if (
            self._bleu(original, generated, version=4) >= self.validation_configs.bleu_4_threshold and 
            self._bleu(original, generated, version=3) >= self.validation_configs.bleu_3_threshold and 
            self._bleu(original, generated, version=2) >= self.validation_configs.bleu_2_threshold and
            self._meteor(original, generated) >= self.validation_configs.meteor_threshold and
            self._cosine_similarity(original, generated) >= self.validation_configs.cosine_similarity_threshold
        ):
            return True
        else:
            return False