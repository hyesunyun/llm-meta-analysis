from model import Model
from openai import OpenAI

class GPT35(Model):
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(organization='org-amDbJ4wMNLPWA2hhgt3UdF7k',)

    def generate_output(self, input: str) -> str:
        """
        This method generates the output given the input

        :param input: input to the model

        :return output of the model
        """
        pass