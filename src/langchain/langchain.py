

class LangChainConnector():
    def __init__(self, prompt_configs) -> None:
        self.prompt_configs = prompt_configs

    def get_most_common_rows(self, row):
        number_of_most_common = self.prompt_configs.number_of_most_common

        if number_of_most_common > 0:
            pass
        else:
            return ""