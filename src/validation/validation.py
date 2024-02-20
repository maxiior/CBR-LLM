from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from langchain_community.embeddings import GPT4AllEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


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

    def validate(self, original, experiment_name):
        generated = pd.read_csv(f'./logs/{experiment_name}_results.csv')['response'].to_lsit()
        original_steps = original['steps'].to_list()
        
        x = 0

        for i, j in zip(original_steps, generated):
            if (
                self._bleu(i, j, version=4) >= self.validation_configs.bleu_4_threshold and 
                self._bleu(i, j, version=3) >= self.validation_configs.bleu_3_threshold and 
                self._bleu(i, j, version=2) >= self.validation_configs.bleu_2_threshold and
                self._meteor(i, j) >= self.validation_configs.meteor_threshold and
                self._cosine_similarity(i, j) >= self.validation_configs.cosine_similarity_threshold
            ):
                x += 1
        
        return x, len(original_steps)