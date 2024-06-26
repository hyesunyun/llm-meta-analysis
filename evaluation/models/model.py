from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def generate_output(self, input: str, max_new_tokens: int) -> str:
        """
        This method must be overridden

        :abstract

        """
        pass

    @abstractmethod
    def get_context_length(self) -> int:
        """
        This method must be overridden

        :abstract

        """
        pass
    
    @abstractmethod
    def encode_text(self, text: str) -> str:
        """
        This method encodes the text

        :param text: text to encode

        :return encoded text
        """
        pass
