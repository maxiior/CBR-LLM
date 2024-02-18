from ..langchain import LangChainConnector
import json

class PromptCreator():
    def __init__(self, experiment_configs) -> None:
        self.prompt_configs = experiment_configs.prompt
        self.langchain_connector = LangChainConnector(self.prompt_configs)

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