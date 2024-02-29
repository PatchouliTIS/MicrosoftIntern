import os
import glob
from typing import Dict
from code_generator import CodeGenerator


class ComponentGenerator(CodeGenerator):
    def __init__(self, prompt_template_path: str) -> None:
        super().__init__(prompt_template_path)
        self.refs = self.load_refs(
            os.path.join(os.path.dirname(__file__), "..", "refs", "component_docs")
        )


    def generate(self, description: str) -> Dict[str, str]:
        """Given description of a component, generate component code and component_spec configuration

        Args:
            description (str): description of the component

        Returns:
            Dict[str, str]: dictionary containing the generated code and component_spec, <file_name>: <file_content>
        """
        prompt = self.format_prompt(description)
        # call codex to generate the code, TODO: implement the call to codex
        response = self.codex.generate(prompt)
        return self.parse_response(response)

    def load_refs(self, ref_folder: str) -> Dict[str, str]:
        """walks through the ref_folder and loads the references into a dictionary

        Args:
            ref_folder (str): folder containing the references

        Returns:
            Dict[str, str]: dictionary containing the references
        """
        return {os.path.basename(file_path): open(file_path, 'r').read() for file_path in glob.glob(os.path.join(ref_folder, '*'))}

    def format_prompt(self, description: str) -> str:
        """format the prompt with the description

        Args:
            description (str): description of the component

        Returns:
            str: formatted prompt
        """
        return self.prompt_template.replace("{{description}}", description)
    
    def parse_response(self, response: str) -> Dict[str, str]:
        """parse the response into code and component_spec

        Args:
            response (str): response from codex

        Returns:
            Dict[str, str]: dictionary containing the code and component_spec
        """
        # TODO: implement the parsing logic
        pass