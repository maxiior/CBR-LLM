

class LangChainConnector():
    def __init__(self, prompt_configs) -> None:
        self.prompt_configs = prompt_configs

    def get_most_common_rows(self, row):
        number_of_most_common = self.prompt_configs.number_of_most_common

        if number_of_most_common > 0:
            text = f'name: {row["name"]} ; ingredients: {row["ingredients"]} ; preparation: {row["steps"]}'
        else:
            return ""