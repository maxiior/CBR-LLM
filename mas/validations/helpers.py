
import pandas as pd
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
import os


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





