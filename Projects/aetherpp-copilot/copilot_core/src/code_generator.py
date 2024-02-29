
import abc
from typing import Dict


class CodeGenerator(metaclass=abc.ABCMeta):
    def __init__(self, prompt_template_path) -> None:
        self.prompt_template = self.load_prompt_template(prompt_template_path)

    @abc.abstractmethod
    def generate(self, description: str) -> Dict[str, str]:
        pass

    def load_prompt_template(self, prompt_template_path: str) -> str:
        with open(prompt_template_path, 'r') as file:
            return file.read()