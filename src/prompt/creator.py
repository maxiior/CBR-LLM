import json
import numpy as np
import pandas as pd
import os

class PromptCreator():
    def __init__(self, experiment_configs) -> None:
        self.prompt_configs = experiment_configs.prompt
        self.model_configs = experiment_configs.model
        self.save_files_prefix = experiment_configs.save_files_prefix

    def _get_prompt(self, prompt_name: str):
        prompts = {}
        with open('prompt/prompts.json', 'r') as file:
            prompts = json.load(file)
        return prompts[prompt_name]
        
    def create_prompt(self, row):
        prompt_tags = self.prompt_configs.tags
        prompt = self._get_prompt(self.prompt_configs.name)

        most_common = self.langchain_connector.get_most_common_rows(row)

        for i in prompt_tags:
            if i.value not in prompt:
                raise ValueError(f'Promt does not contain {i} tag.')

        for i in prompt_tags:
            if i.source == 'dataset':
                prompt = prompt.replace(i.value, row[i.column])
            elif i.source == 'langchain' and most_common != "":
                prompt = prompt.replace(i.value, most_common)

        return prompt
    
    def prepare_prompts(self, masked_dataset, original_dataset, file_name, casebase):
        number_of_examples = self.prompt_configs.number_of_examples
        random = self.prompt_configs.random

        file_name = f"{self.save_files_prefix}_{file_name}.csv"

        if os.path.exists(f'../dataset/prompts/{file_name}'):
            print('INFO - prompts already exists for this dataset.')
            return pd.read_csv(f"../dataset/prompts/{file_name}")
        else:
            original_dataset = original_dataset.to_numpy()
            prompts = []

            for idx, i in enumerate(masked_dataset.to_numpy()):
                if not random:
                    original = f"name: {original_dataset[idx][1]} ; ingredients: {original_dataset[idx][3]}"

                    examples = casebase.similarity_search(query=original, k=number_of_examples)

                    prepared_examples = " ; ".join([i.page_content for i in examples])
                else:
                    examples = np.random.choice(original_dataset.shape[0], size=number_of_examples)
                    examples = original_dataset[examples, :]
                    prepared_examples = " ; ".join([f"name: {j[1]} ; ingredients: {j[3]} ; preparation: {j[2]}" for j in examples])

                prompts.append(f"{prepared_examples} ; name: {i[1]} ; ingredients: {i[3]} ; preparation:")
            
            df = pd.DataFrame()
            df['id'] = [i[0] for i in original_dataset]
            df['input'] = prompts

            self._save_prompts(df, file_name)
            return df
    
    def _save_prompts(self, df, file_name):
        df.to_csv(f"../dataset/prompts/{file_name}", index=False)
        print('INFO - prompts csv has been made.')