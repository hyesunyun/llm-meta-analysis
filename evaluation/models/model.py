from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def generate_output(self, input: str) -> str:
        """
        This method must be overridden

        :abstract

        """
        pass
