from .interfaces import Dataset
import numpy as np
from ast import literal_eval
import re

class DataCreator(Dataset):
    def __init__(self, experiment_configs):
        super().__init__(experiment_configs)

    def _validate_symbols(self):
        not_allowed = ['/', "'", '!', '"', '%', ')', '@', 
                       '(', '~', '#', '[', '=', ']', '+', 
                       '?', '`', '}', '{', '>', '^', '<', 
                       '_', '\\', '$', '|', '*', ':', ';', '.']
        def fun(element):
            try:
                element = literal_eval(element)
            except:
                pass

            if type(element) == list:
                for i in element:
                    if len([j for j in not_allowed if j in i]) != 0:
                        return False
                return True
            elif type(element) == str:
                return len([i for i in not_allowed if i in element]) == 0
            else:
                return True
        return fun
    
    def _allowed_signs_processing(self, text):
        text = text.replace(f" ,", f",")
        text = text.replace(f" .", f".")
        return " ".join(text.split())

    def _concatenate_steps(self, steps):
        steps = literal_eval(steps)
        text = " ".join([f"{i.capitalize()}." for i in steps])
        return self._allowed_signs_processing(text)

    def _concatenate_ingredients(self, ingredients):
        ingredients = literal_eval(ingredients)
        text = ", ".join(ingredients)
        return self._allowed_signs_processing(text)
    
    def _processing_name(self, name):
        name = self._allowed_signs_processing(name)
        return name.lower()

    def _validate_sentence_length(self, steps):
        steps = literal_eval(steps)
        for step in steps:
            if len(step) <= 3:
                return False
        return True
    
    def _validate_words_number_in_row(self, name, steps, ingredients):
        pattern = re.compile('[a-zA-Z\s]+')
        text = pattern.findall(f"{name} {steps} {ingredients}")

        if len(" ".join(text).split()) <= 200:
            return True
        else:
            return False

    def _preprocessing_dataframe(self, df):
        df = df.dropna()
        df = df[df.n_steps<=12][df.n_steps>=5]
        df = df[df.n_ingredients<=10][df.n_ingredients>=4]
        df = df.drop("n_steps", axis=1)
        df = df.drop("n_ingredients", axis=1)

        df = df[np.vectorize(self._validate_symbols())(df.name)]
        df = df[np.vectorize(self._validate_symbols())(df.steps)]
        df = df[np.vectorize(self._validate_symbols())(df.ingredients)]

        df = df[np.vectorize(self._validate_words_number_in_row)(df.name, df.steps, df.ingredients)]
        
        df = df[np.vectorize(self._validate_sentence_length)(df.steps)]

        df.name = df.name.apply(lambda x: self._processing_name(x))
        df.steps = df.steps.apply(lambda x: self._concatenate_steps(x))
        df.ingredients = df.ingredients.apply(lambda x: self._concatenate_ingredients(x))

        return df