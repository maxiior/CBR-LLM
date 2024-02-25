from configs import Configs
from prompt import PromptCreator
from data import DataCreator
from models import ModelManager
from validation import Validation
from experiments import Experiment
from casebase import CBRDatabase

class LlamaCPPMock():
    def __init__(self, model_path="models/llama-13b-hf_q8_0.gguf", n_gpu_layers=41, n_batch=1024, n_ctx=2048) -> None:
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.model = None

    def send_request(self, X, break_word:str = " name:") -> str:
        return X

    def get_info(self) -> str:
        return {
            "model":"LLama-13b-hf-q8_0",
            "model_path":self.model_path,
            "n_gpu_layers":self.n_gpu_layers,
            "n_batch":self.n_batch,
            "n_ctx":self.n_ctx,
        }

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
            experiment = Experiment(experiment_configs=configs)

            datasets = data_creator.create_structured_datasets()
            # model = model_manager.get_model()
            model = LlamaCPPMock()

            if mode == "auto":
                cbr_dataset = data_creator.prepare_cbr_dataset(datasets['cbr_database'], ['name', 'ingredients'])
                
                metadata = CBRDatabase.create_metadatas(list(range(len(cbr_dataset))))
                casebase = CBRDatabase(texts=cbr_dataset, db_directory="dataset_chroma_db", metadatas=metadata)

                # docs = db.similarity_search(query="apple", k=4)

                masked_dataset, original_dataset = data_creator.prepare_masked_dataset(datasets['cbr_database'])
                prompts = prompt_creator.prepare_prompts(masked_dataset, original_dataset, file_name='cbr_database', casebase=casebase)

                # experiment.run(model, prompts, dataset_name='small_validation_pe')









                # validation.validate(original_dataset, 'small_validation_pe')
                

                
            # elif mode == "manual":
            #     user_inputs = self._get_user_inputs()
            # else:
            #     raise ValueError(f'{mode} mode does not exists.')
        

wf = Workflow()

wf.run()