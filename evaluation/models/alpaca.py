from .model import Model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class Alpaca(Model):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def get_context_length(self) -> int:
        return 4096
    
    def encode_text(self, text: str) -> str:
        """
        This method encodes the text

        :param text: text to encode

        :return encoded text
        """
        return self.tokenizer.encode(text)

    def __load_model(self):   
        ALPACA_MODEL_PATH = os.getenv('ALPACA_MODEL_PATH')
        model = AutoModelForCausalLM.from_pretrained(ALPACA_MODEL_PATH, device_map="auto")
        return model

    def __load_tokenizer(self):
        ALPACA_MODEL_PATH = os.getenv('ALPACA_MODEL_PATH')
        tokenizer = AutoTokenizer.from_pretrained(ALPACA_MODEL_PATH)
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
            return self.tokenizer.decode(result[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            print("[ERROR]", e)