
from .interfaces import Model
from langchain_community.llms import LlamaCpp
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Tokenizer


class GPT2(Model):
    def __init__(self, max_length=1024, num_beams=3) -> None:

        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.num_beams = num_beams

        super().__init__()

    def send_request(self, X:list[str]) -> str:

        encoded_input = self.tokenizer(X, padding=True, return_tensors='pt').to('cuda')
        output = self.model.generate(**encoded_input, max_length=self.max_length, num_beams=self.num_beams, num_return_sequences=1)
        decoded = self.tokenizer.batch_decode(output)
        

        return decoded

    def get_info(self) -> str:
        return {
            "model":"GPT-2",
            "max_length":self.max_length,
            "max_length":self.num_beams
        }


