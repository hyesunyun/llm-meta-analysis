from .model import Model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class BioMistral(Model):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def get_context_length(self) -> int:
        return 2048
    
    def encode_text(self, text: str) -> str:
        """
        This method encodes the text

        :param text: text to encode

        :return encoded text
        """
        return self.tokenizer(text, return_tensors="pt").input_ids

    def __load_model(self):
        # fine-tuned on biomedical texts but only to context of 2048 tokens
        # also have only been evaluated on biomedical tasks that are multiple choice questions 
        # after supervised fine tuning so not sure it's true performance on true zero shot open ended generation     
        model = AutoModelForCausalLM.from_pretrained("BioMistral/BioMistral-7B", device_map="auto")
        return model

    def __load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
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
                result = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True) # same as mistral
            return self.tokenizer.decode(result[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            print("[ERROR]", e)