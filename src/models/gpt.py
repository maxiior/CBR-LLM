from openai import OpenAI
from dotenv import load_dotenv
from interfaces import Model
import os


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class GPTConnector(Model):
    def __init__(self, model_configs, api_key=OPENAI_API_KEY) -> None:
        self.model_configs = model_configs
        self.client = OpenAI(api_key=api_key)
      
    def send_request(self, prompt):
        return self.client.chat.completions.create(
            model=self.model_configs.name,
            prompt=prompt,
            temperature=self.model_configs.params.temperature,
            response_format=self.model_configs.params.response_format
        )