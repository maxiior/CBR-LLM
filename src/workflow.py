from .configs import Configs
from .prompt import PROMPT_1, PROMPT_2, PROMPT_3, PromptCreator
from .models import GPTConnector

class Workflow():
    def __init__(self) -> None:
        self.configs = Configs.get()
        self.prompt_crator = PromptCreator(self.configs)
        self.model = GPTConnector()

    def _get_user_inputs(self):
        user_inputs = {}
        for i in self.configs.request_body.prompt_tags:
            user_inputs[i] = input(i)

    def run(self):
        mode = self.configs.workflow.mode

        if mode == "auto":
            pass
        elif mode == "manual":
            user_inputs = self._get_user_inputs()
            prompt = self.prompt_crator(user_inputs, PROMPT_1)
            request = self.model.send_prompt(prompt)


        else:
            raise ValueError(f'{mode} mode does not exists.')