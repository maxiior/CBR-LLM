from configs import Configs
from prompt import PromptCreator
from data import DataCreator
from models import ModelManager
from validation import Validation

class Workflow():
    def __init__(self) -> None:
        self.configs = Configs.get_main_configs()
        self.experiments_configs = Configs.get_experiments_configs()

    def _get_user_inputs(self):
        user_inputs = {}
        for i in self.configs.request_body.prompt_tags:
            user_inputs[i] = input(i)

    def run(self):
        mode = self.configs.workflow.mode

        for configs in self.experiments_configs:
            data_creator = DataCreator(experiment_configs=configs)
            prompt_creator = PromptCreator(experiment_configs=configs)
            model_manager = ModelManager(experiment_configs=configs)
            validation = Validation(experiment_configs=configs)

            datasets = data_creator.create_structured_datasets()
            model = model_manager.get_model()

            if mode == "auto":
                masked_dataset, original_dataset = data_creator.prepare_masked_dataset(datasets['train'])
                
                for idx, row in masked_dataset.iterrows():
                    prompt = prompt_creator.create_prompt(row)
                    response = model.send_request(prompt)

                    result = validation.validate(original_dataset.iloc[idx], response)

                    if result:
                        #add to langchain
                        pass

                

                
            # elif mode == "manual":
            #     user_inputs = self._get_user_inputs()
            #     prompt = self.prompt_crator(user_inputs)
            #     request = self.model.send_prompt(prompt)


            else:
                raise ValueError(f'{mode} mode does not exists.')
        

wf = Workflow()

wf.run()