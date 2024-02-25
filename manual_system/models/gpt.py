from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer

class GPT2():
    def __init__(self, model_name) -> None:
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left" 
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_number_of_tokens(self, prompt):
        return len(self.tokenizer.encode(prompt))

    def send_request(self, prompt):
        encoded_input = self.tokenizer(prompt, padding=True, return_tensors='pt')
        output = self.model.generate(**encoded_input, max_length=1024, num_beams=2, num_return_sequences=1)
        decoded = self.tokenizer.batch_decode(output)

        return decoded