from typing import Dict
import json

class PromptCreator():
    def __init__(self, experiment_configs) -> None:
        self.prompt_configs = experiment_configs.prompt

    def _get_prompt(self, prompt_name: str):
        prompts = {}
        with open('prompt/prompts.json', 'r') as file:
            prompts = json.load(file)
        return prompts[prompt_name]
        
    def create_prompt(self, inputs: Dict):
        prompt_tags = self.prompt_configs.tags
        prompt = self._get_prompt(self.prompt_configs.file_name)

        for i in prompt_tags:
            if i not in prompt:
                raise ValueError(f'Promt does not contain {i} tag.')

        for i in prompt_tags:
            prompt = prompt.replace(i, inputs[i])

        return prompt