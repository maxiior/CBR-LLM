import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
import os

class CBRDatabase:
    RECIPE_ID_KEY = "recipe_id"

    def __init__(self, texts, metadatas:list=None, db_directory="dataset_chroma_db") -> None:
        embedding_function = GPT4AllEmbeddings()
        
        if os.path.isdir(db_directory):
            self.db = Chroma(persist_directory=db_directory, embedding_function=embedding_function)
        else:
            self.db = Chroma.from_texts(texts=texts, embedding=embedding_function, metadatas=metadatas, persist_directory=db_directory)

    def add_examples(self, texts, metadatas:list = None):
        self.db.add_texts(texts, metadatas)

    def similarity_search(self, **kwargs):
        res = self.db.similarity_search(**kwargs)
        return res

    @staticmethod
    def create_metadatas(recipe_indices:list):
        return [{CBRDatabase.RECIPE_ID_KEY:e} for e in recipe_indices]