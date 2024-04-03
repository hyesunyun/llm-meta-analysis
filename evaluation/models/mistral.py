from .model import Model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Mistral(Model):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def get_context_length(self) -> int:
        return 32768
    
    def encode_text(self, text: str) -> str:
        """
        This method encodes the text

        :param text: text to encode

        :return encoded text
        """
        return self.tokenizer.encode(text)
        
    def __load_model(self): # context window size: 32k tokens but 8k tokens is recommended for best performance
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", device_map="auto"
        )
        return model

    def __load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        # tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
        return tokenizer

    def generate_output(self, input: str, max_new_tokens: int) -> str:
        """
        This method generates the output given the input. Uses chat template for input.

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate

        :return output of the model
        """
        try:
            chat = [
                {"role": "user", "content": input},
            ]
            encoded = self.tokenizer.apply_chat_template(chat, return_tensors="pt")
            model_inputs = encoded.to(self.device)
            with torch.no_grad():
                result = self.model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=True) # used the same parameters as the example in the hugginface model card: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
            return self.tokenizer.decode(result[0, model_inputs.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            print("[ERROR]", e)




