from .model import Model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Mixtral(Model):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def load_model(self): # context window size: 32k tokens but 8k tokens is recommended for best performance
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1", device_map="auto"
        )
        return model

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
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