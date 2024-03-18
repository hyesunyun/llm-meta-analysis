from .model import Model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Llama2(Model): 
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def load_model(self): # context window size: 32k tokens - special version of Llama2 7B
        # can use flash attention with trust_remote_code=True
        model = AutoModelForCausalLM.from_pretrained(
            "togethercomputer/Llama-2-7B-32K-Instruct",
            trust_remote_code=False, # turning off flash attention due to environment issues
            torch_dtype=torch.float16
        ).to(self.device)
        return model

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/Llama-2-7B-32K-Instruct")
        return tokenizer

    def generate_output(self, input: str, max_new_tokens: int) -> str:
        """
        This method generates the output given the input

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate

        :return output of the model
        """
        try:
            inputs = self.tokenizer(input, return_tensors="pt").to(self.device)
            with torch.no_grad():
                result = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(result[0], skip_special_tokens=True)
        except Exception as e:
            print("[ERROR]", e)