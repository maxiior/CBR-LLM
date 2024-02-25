
from .interfaces import Model
from langchain_community.llms import LlamaCpp


class LlamaCPP(Model):
    def __init__(self, model_path="models/llama-13b-hf_q8_0.gguf", n_gpu_layers=41, n_batch=1024, n_ctx=2048) -> None:
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.n_ctx = n_ctx

        self.model = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            verbose=True,
        )
        super().__init__()

    def send_request(self, X, break_word:str = " name:") -> str:
        tok_len = self.model.get_num_tokens(X)
        # print(tok_len)
        if tok_len > 1550:
            raise Exception(f"Request exceeds prefelable 1500 tokens. Has: {tok_len}")
        prev = ""
        res = ""
        for token in self.model.stream(X, echo=False):
            res += token
            if break_word == prev+token:
                res = res.replace(" ; name:", "")
                res = res.replace(";name:", "")
                res = res.replace(" ;name:", "")
                res = res.replace("; name:", "")
                res = res.replace(" name:", "")
                res = res.replace("name:", "")
                break
            if "\n" in token:
                res = res.replace("\n", "")
                break
            prev = token
        
        res = res.replace("  ;  ", "")
        res = res.replace("  ; ", "")
        res = res.replace("  ;", "")
        res = res.replace(" ; ", "")
        res = res.replace("; ", "")
        res = res.replace(" ;", "")
        res = res.replace(";", "")
        return res

    def get_info(self) -> str:
        return {
            "model":"LLama-13b-hf-q8_0",
            "model_path":self.model_path,
            "n_gpu_layers":self.n_gpu_layers,
            "n_batch":self.n_batch,
            "n_ctx":self.n_ctx,
        }


