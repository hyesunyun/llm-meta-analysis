from model import Model
from openai import OpenAI

class GPT35(Model):
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(organization='org-amDbJ4wMNLPWA2hhgt3UdF7k',)

    def generate_output(self, input: str, temperature: str = 1) -> str:
        """
        This method generates the output given the input

        :param input: input to the model

        :return output of the model
        """
        try:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for conducting meta-analyses of randomized controlled trials."},
                    {"role": "user", "content": input}
                ],
                # TODO: currently set as default but should figure out temperature/top_p parameters
                # https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683
                temperature=temperature,
                top_p=1,
            )
        except Exception as e:
            print(e)
            return "Error: GPT-3.5 API call failed."

        return completion.choices[0].message.content