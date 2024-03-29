import os
import glob
import re
import uuid
from typing import Dict
from codex_generator import CodeXGenerator
from codex_getter import CodexGetter
from utils.helper import write_files


class CodeXComponentCreator(CodeXGenerator):
    def __init__(self, prompt_template_path: str) -> None:
        super().__init__(prompt_template_path)
        self.refs = self.load_refs(
            os.path.join(os.path.dirname(__file__), "..", "refs", "component_docs")
        )


    def generate(self, description: str, path: str = ".") -> Dict[str, str]:
        """Given description of a component, generate component code and component_spec configuration

        Args:
            description (str): description of the component

        Returns:
            Dict[str, str]: dictionary containing the generated code and component_spec, <file_name>: <file_content>
        """
        prompt = self.format_prompt(description)
        print(f"prompt: {prompt}")
        # call codex to generate the code, TODO: implement the call to codex
        codex_getter = CodexGetter(
            prompt,
            temperature=0.7,
            max_tokens=1024,
            top_p=1.0,
            streaming=False,
            token=self.token,
        )
        codex_getter.run()
        file_contents, folder_name = self.parse_response(codex_getter.messages[0])
        file_paths = write_files(file_contents, os.path.join(path, folder_name))
        return file_paths


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
        formatted_prompt = self.prompt_template.replace("{{description}}", description)
        formatted_prompt = formatted_prompt.replace("{{refs}}", "\n".join(self.refs.values()))
        return formatted_prompt
    
    def parse_response(self, response: str) -> Dict[str, str]:
        """parse the response into code and component_spec

        Args:
            response (str): response from codex

        Returns:
            Dict[str, str]: dictionary containing the code and component_spec
        """
        print(f"response: {response}")
        pattern = r'file name: (.*?)\n```(.*?)\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        file_contents = {file_name.strip(): file_content.strip() for file_name, _, file_content in matches}
        # try to get name from component_spec.yaml
        try:
            component_spec = file_contents["component_spec.yaml"]
            name = re.search(r'name: (.*?)\n', component_spec).group(1)
        except:
            print(f"failed to parse name, generating a random name for the component...")
            name = str(uuid.uuid4())
        return file_contents, name


if __name__ == "__main__":
    prompt_template_path = os.path.join(
        os.path.dirname(__file__), "..", "prompt_templates", "component_generator.txt"
    )
    print(f"prompt_template_path: {prompt_template_path}")
    component_generator = CodeXComponentCreator(prompt_template_path)
    description = "compare difference btw two input files"
    print(component_generator.generate(description))
