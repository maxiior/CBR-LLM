from .configs import Configs
from .prompt import PromptCreator
from .models import GPTConnector
from .data import DataCreator

class Workflow():
    def __init__(self) -> None:
        self.configs = Configs.get_main_configs()
        self.experiments_configs = Configs.get_experiments_configs()

        self.prompt_crator = PromptCreator(self.configs)
        self.model = GPTConnector()

    def _get_user_inputs(self):
        user_inputs = {}
        for i in self.configs.request_body.prompt_tags:
            user_inputs[i] = input(i)

    def run(self):
        mode = self.configs.workflow.mode

        for configs in self.experiments_configs:
            dc = DataCreator(experiment_configs=configs)
            dataset = dc.create_structured_datasets()

            if mode == "auto":
                masked_dataset, original_dataset = dc.prepare_masked_dataset(dataset['train'])
                
                
            elif mode == "manual":
                user_inputs = self._get_user_inputs()
                prompt = self.prompt_crator(user_inputs)
                request = self.model.send_prompt(prompt)


        else:
            raise ValueError(f'{mode} mode does not exists.')