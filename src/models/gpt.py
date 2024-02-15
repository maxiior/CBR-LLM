from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class GPTConnector():
    def __init__(self, configs, model="gpt-3.5-turbo-instruct", api_key=OPENAI_API_KEY) -> None:
        self.configs = configs
        self.model = model
        self.client = OpenAI(api_key=api_key)
      
    def send_prompt(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=self.configs.request_body.temperature,
            response_format=self.configs.request_body.response_format
        )
        return response