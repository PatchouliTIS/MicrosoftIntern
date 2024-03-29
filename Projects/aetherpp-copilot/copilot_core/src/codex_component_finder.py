import os
import glob
import re
import uuid
from typing import Dict, List
from codex_generator import CodeXGenerator
from codex_getter import CodexGetter
from component import Component


class CodexComponentFinder(CodeXGenerator):
    def __init__(self, prompt_template_path) -> None:
        super().__init__(prompt_template_path)


    def generate(self, description: str, name_to_componnets: Dict[str, Component], top_k: int) -> List[str]:
        prompt = self.format_prompt(description, name_to_componnets, top_k)
        print(f"prompt: {prompt}")
        # call codex to find components
        codex_getter = CodexGetter(
            prompt,
            temperature=0.7,
            max_tokens=10240,
            top_p=1.0,
            streaming=False,
            token=self.token,
        )
        codex_getter.run()
        component_names = self.parse_response(codex_getter.messages[0])
        return component_names
    
    def format_prompt(self, description: str, name_to_componnets: Dict[str, Component], top_k: int) -> str:
        """format the prompt with the description and component contents
        Args:
            description (str): description of the component
            name_to_componnets (Dict[str, Component]): dictionary containing the name to component mapping
        Returns:
            str: formatted prompt
        """
        name_desc_str_list = [f"name: {name}, display_name: {component.display_name}, description: {component.description}" for name, component in name_to_componnets.items()]
        formatted_prompt = self.prompt_template.replace("{{description}}", description)
        formatted_prompt = formatted_prompt.replace("{{top_k}}", str(top_k))
        formatted_prompt = formatted_prompt.replace("{{components}}", "\n".join(name_desc_str_list))
        return formatted_prompt
    
    def parse_response(self, response: str) -> List[str]:
        """parse the response from codex
        Args:
            response (str): response from codex
        Returns:
            List[str]: list of component names
        """
        print(f"response: {response}")
        component_names = [line.strip("- ") for line in response.split("\n") if line.startswith("- ")]
        return component_names
