from .model import Model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import hf_olmo

class Olmo(Model):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def get_context_length(self) -> int:
        return 2048

    def load_model(self): # context window size: 32k tokens but 8k tokens is recommended for best performance
        model = AutoModelForCausalLM.from_pretrained(
            "allenai/OLMo-7B-Instruct", device_map="auto"
        )
        return model

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-Instruct")
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
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
            with torch.no_grad():
                result = self.model.generate(inputs, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(result[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            print("[ERROR]", e)