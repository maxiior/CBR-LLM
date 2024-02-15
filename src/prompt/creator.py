from typing import Dict

class PromptCreator():
    def __init__(self, configs) -> None:
        self.configs = configs

    def create_prompt(self, inputs: Dict, prompt: str):
        prompt_tags = self.configs.request_body.prompt_tags

        for i in prompt_tags:
            if i not in prompt:
                raise ValueError(f'Promt does not contain {i} tag.')

        for i in prompt_tags:
            prompt = prompt.replace(i, inputs[i])

        return prompt

            
        
