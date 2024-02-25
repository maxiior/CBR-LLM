from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from langchain_community.embeddings import GPT4AllEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import statistics


class Validation():
    def __init__(self) -> None:
        self.tokenizer = GPT4AllEmbeddings()

    def _cosine_similarity(self, original:str, generated:str) -> float:
        X = self.tokenizer.embed_query(str(generated))
        Y = self.tokenizer.embed_query(str(original))
        score = cosine_similarity([X], [Y])[0][0]
        return round(score, 6)

    def _bleu(self, original:str, generated:str, version:int=4) -> float:
        weights = [0.25, 0.25, 0.25, 0.25]
        if version == 3:
            weights = [0.33, 0.33, 0.33]
        elif version == 2:
            weights = [0.5, 0.5]

        return round(sentence_bleu([original.split()], generated.split(), weights=weights), 6)
    
    def _meteor(self, original:str, generated:str) -> float:
        return round(meteor_score([original.split()], generated.split()), 6)

    def get_avg_metrics(self, original, generated):
        bleu_2, bleu_3, bleu_4, meteor, cos_sim = [], [], [], [], []
        
        for i, j in zip(original, generated):
            bleu_4.append(self._bleu(i, j, version=4))
            bleu_3.append(self._bleu(i, j, version=3))
            bleu_2.append(self._bleu(i, j, version=2))
            meteor.append(self._meteor(i, j))
            cos_sim.append(self._cosine_similarity(i, j))
        
        print(f'bleu_2: {statistics.mean(bleu_2)}')
        print(f'bleu_3: {statistics.mean(bleu_3)}')
        print(f'bleu_4: {statistics.mean(bleu_4)}')
        print(f'meteor: {statistics.mean(meteor)}')
        print(f'cos_sim: {statistics.mean(cos_sim)}')