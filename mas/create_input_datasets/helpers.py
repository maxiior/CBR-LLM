
import pandas as pd
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
import os


def validate_datestet(df1:pd.DataFrame, df2:pd.DataFrame):
    return np.all(df1.target_recipe.values == df2.target_recipe.values)

def compose(factual:pd.DataFrame, masked:"pd.Row") -> str:
    comp = [f"name: {e['name']} ; ingredients: {e['ingredients']} ; preparation:{e['steps']}" for i, e in factual.iterrows()]
    comp = comp + [f"name: {masked['name']} ; ingredients: {masked['ingredients']} ; preparation:"]
    return " ; ".join(comp)

def str_rep_no_steps(df_recipes) -> str:
    comp = f"name: {df_recipes['name']} ; ingredients: {df_recipes['ingredients']} ; preparation:"
    return comp

def str_rep(df_recipes) -> str:
    comp = [f"name: {e['name']} ; ingredients: {e['ingredients']} ; preparation:{e['steps']}" for i, e in df_recipes.iterrows()]
    return comp

def str_rep_vdb(df_recipes) -> str:
    comp = [f"name: {e['name']} ; ingredients: {e['ingredients']} ; preparation:" for i, e in df_recipes.iterrows()]
    return comp

def load_all_datasets(datasets_path:str="datasets/") -> tuple:
    additional_cbr_1 = pd.read_csv(f"{datasets_path}_additional_cbr_1.csv")
    additional_cbr_2 = pd.read_csv(f"{datasets_path}_additional_cbr_2.csv")
    additional_validation = pd.read_csv(f"{datasets_path}_additional_validation.csv")
    cbr_augmentation = pd.read_csv(f"{datasets_path}_cbr_augmentation.csv")
    cbr_database = pd.read_csv(f"{datasets_path}_cbr_database.csv")
    model_training = pd.read_csv(f"{datasets_path}_model_training.csv")
    validation = pd.read_csv(f"{datasets_path}_validation.csv")

    return additional_cbr_1, additional_cbr_2, additional_validation, cbr_augmentation, cbr_database, model_training, validation


class CBRDatabase:
    RECIPE_ID_KEY = "recipe_id"

    def __init__(self, texts = None, metadatas:list = None, db_directory="dataset_chroma_db", initialize_new=False) -> None:
        embedding_function = GPT4AllEmbeddings()
        
        if initialize_new:
            if os.path.isdir(db_directory): raise Exception("Provided directory exists")
            self.db = Chroma.from_texts(texts = texts, embedding = embedding_function, metadatas=metadatas, persist_directory=db_directory)
        else:
            if texts or metadatas: raise Exception("You are trying to load exising database, not crate new.")
            self.db = Chroma(persist_directory=db_directory, embedding_function=embedding_function)

    def add_examples(self, texts, metadatas:list = None):
        self.db.add_texts(texts, metadatas)

    def similarity_search(self, **kwargs):
        """query=query, k=4, filter={'test':True}"""
        res = self.db.similarity_search(**kwargs)
        return res

    @staticmethod
    def create_metadatas(recipe_indices:list, batch_metadata:dict = None):
        metadatas = [{CBRDatabase.RECIPE_ID_KEY:e} for e in recipe_indices]
        if batch_metadata:
            for d in metadatas:
                d.update(batch_metadata)
        return metadatas





