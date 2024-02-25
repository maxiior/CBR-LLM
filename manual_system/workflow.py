from data import DataCreator
from models import GPT2
from casebase import CBRDatabase
import pandas as pd


class Workflow():
    MAX_NUMBER_OF_TOKENS = 750

    def __init__(self) -> None:
        pass

    def _get_user_inputs(self):
        user_inputs = {}
        for i in ['name', 'ingredients']:
            user_inputs[i] = input(f'{i}: ')
        return user_inputs

    def run(self):
        data_creator = DataCreator(
            source_file_name='RAW_recipes.csv',
            save_files_prefix='user',
            columns_to_extract=["id","name","n_steps","steps","ingredients","n_ingredients"],
            proportions={
                "cbr_database": 0.188
            }
        )

        datasets = data_creator.create_structured_datasets()
        model = GPT2(model_name='gpt2')

        cbr_dataset, cbr_ids = data_creator.prepare_cbr_dataset(datasets['cbr_database'], ['name', 'ingredients'])
        metadata = CBRDatabase.create_metadatas(cbr_ids)
        casebase = CBRDatabase(texts=cbr_dataset, db_directory="dataset_chroma_db", metadatas=metadata)

        while True:
            user_inputs = self._get_user_inputs()
            cbr_prompt = f'name: {user_inputs["name"]} ; ingredients: {user_inputs["ingredients"]}'

            docs = casebase.similarity_search(query=cbr_prompt, k=10)

            j = 10

            while True:
                llm_prompt = ''
                
                for idx, i in enumerate(docs):
                    recipe_id = i.metadata['recipe_id']

                    row = datasets['cbr_database'].loc[datasets['cbr_database']['id'] == recipe_id]

                    name = row.iloc[0]['name']
                    ingredients = row.iloc[0]['ingredients']
                    steps = row.iloc[0]['steps']
                    
                    llm_prompt += f'name: {name} ; '
                    llm_prompt += f'ingredients: {ingredients} ; '
                    llm_prompt += f'preparation: {steps} ; '

                    if idx == j - 1:
                        break
                
                llm_prompt += cbr_prompt + ' ; preparation: '

                if model.get_number_of_tokens(llm_prompt) <= self.MAX_NUMBER_OF_TOKENS:
                    break
                else:
                    j -= 1

            print(f"HINTS: {j}")

            response = model.send_request(llm_prompt)[0]
            response = response[len(llm_prompt):]
            
            response_name = response.split('name:')
            response_ingredients = response.split('ingredients:')

            if len(response_name[0]) < len(response_ingredients[0]):
                response = response_name[0]
            else:
                response = response_ingredients[0]
            response = response.replace(';', '').strip()

            response = ".".join(list(dict.fromkeys(response.split("."))))

            print("RESPONSE: ", response)
            user_answare = input("Add to Case-Base? [y/n]: ")

            if user_answare == 'y':
                new_id = max(datasets['cbr_database']['id'].to_list()) + 1
                datasets['cbr_database'].loc[len(datasets)] = [new_id, user_inputs["name"], response, user_inputs["ingredients"]]
                casebase.add_examples([f'name: {user_inputs["name"]} ; ingredients: {user_inputs["ingredients"]}'], metadatas=[{"recipe_id": new_id}])
                data_creator.save_data(pd.DataFrame({'id': [new_id], 'name': [user_inputs["name"]], 'steps': [response], 'ingredients': [user_inputs["ingredients"]]}), dataset_name='cbr_database')
            else:
                print("")
            

        

wf = Workflow()

wf.run()