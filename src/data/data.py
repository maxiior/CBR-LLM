from interfaces import Dataset
import numpy as np
from ast import literal_eval

class DataCreator(Dataset):
    def __init__(self, experiment_configs):
        super().__init__(experiment_configs)

    def _validate_symbols(self):
        not_allowed = ['/', "'", '!', '"', '%', ')', '@', 
                       '(', '~', '#', '[', '=', ']', '+', 
                       '?', '`', '}', '{', '>', '^', '<', 
                       '_', '\\', '$', '|', '*']
        def fun(text):
            return len([i for i in not_allowed if i in text]) == 0
        return fun
    
    def _allowed_signs_processing(self, text):
        text = text.replace(f" ,", f",")
        text = text.replace(f" .", f".")
        return " ".join(text.split())

    def _concatenate_steps(self, steps):
        text = " ".join([f"{i.capitalize()}." for i in steps])
        return self._allowed_signs_processing(text)

    def _concatenate_ingredients(self, ingredients):
        text = ", ".join(ingredients)
        return self._allowed_signs_processing(text)
    
    def _processing_name(self, name):
        name = self._allowed_signs_processing(name)
        return name.capitalize()

    def _preprocessing_dataframe(self, df):
        df = df.dropna()
        df = df[df.n_steps<=12][df.n_steps>=5]
        df = df[df.n_ingredients<=10][df.n_ingredients>=4]
        df = df.drop("n_steps")
        df = df.drop("n_ingredients")

        df = df[np.vectorize(self._validate_symbols())(df)]

        df.name = df.name.apply(lambda x: self._processing_name(x))
        df.steps = df.steps.apply(lambda x: self._concatenate_steps(literal_eval(x)))
        df.ingredients = df.ingredients.apply(lambda x: self._concatenate_ingredients(literal_eval(x)))

        return df