from typing import Dict, List
from codex_generator import CodeXGenerator
from codex_getter import CodexGetter
from component import Component


class CodexReferenceGenerator(CodeXGenerator):
    def __init__(self, prompt_template_path) -> None:
        super().__init__(prompt_template_path)

    def generate(self, context: str, component: Component) -> str:
        prompt = self.format_prompt(context, component)
        print(f"prompt: {prompt}")
        # call codex to generate the reference
        codex_getter = CodexGetter(
            prompt,
            temperature=0.7,
            max_tokens=10240,
            top_p=1.0,
            streaming=False,
            token=self.token,
        )
        codex_getter.run()
        import_line = self._parse_import_line(component.path, component.name)
        reference = self.parse_response(codex_getter.messages[0], component.name)
        return {"import_line": import_line, "reference": reference}

    def format_prompt(self, context: str, component: Component) -> str:
        """format the prompt with the context and component contents
        Args:
            context (str): context of the component
            component (Component): component
        Returns:
            str: formatted prompt
        """
        formatted_prompt = self.prompt_template.replace("{{context}}", context)
        reference_code = f"{component.name}_step = {component.name}("
        for param in component.inputs:
            reference_code += f"{param}=<TBD>, "
        reference_code = reference_code[:-2] + ")"
        formatted_prompt = formatted_prompt.replace("{{reference_code}}", reference_code)
        return formatted_prompt
    
    def parse_response(self, response: str, component_name: str) -> str:
        """parse the response from codex
        Args:
            response (str): response from codex
        Returns:
            str: reference
        """
        print(f"response: {response}")
        # substring start from f"{component.name}_step"
        reference = response.split(f"{component_name}_step")[1]
        reference = f"{component_name}_step{reference}"
       
        return reference
    
    def _parse_import_line(self, component_path: str, component_name: str) -> str:
        """parse the import line from the component path
        Args:
            component_path (str): path to the component
        Returns:
            str: import line
        """
        # example input: D:\\GitSources\\deeprank\\smile\\src\\smile\\components\\common\\concat_files\\component_spec.yaml
        # example output: from smile.components.common import concat_files
        relative_path = component_path.split('src\\')[-1].split('\\')[0:-2]
        relative_path = '.'.join(relative_path)
        import_line = f"from {relative_path} import {component_name}"
        return import_line