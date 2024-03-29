import os
import yaml
import difflib
from codex_component_finder import CodexComponentFinder
from component import Component, ComponentType
from typing import List, Dict

class ComponentManager:
    def __init__(self, component_folder: str) -> None:
        self.component_folder = component_folder
        self.name_to_componnets = self.load_components()
        self.codex_component_finder = CodexComponentFinder(
            os.path.join(os.path.dirname(__file__), "..", "prompt_templates", "component_finder.txt")
        )

    def load_components(self) -> List[Component]:
        """Scan the component_folder and load the components

        Returns:
            List[Component]: List of components
        """
        name_to_componnets: Dict[str, Component] = {}
        # scan *.yaml files in the component_folder
        for root, dirs, files in os.walk(self.component_folder):
            for file in files:
                if file.endswith(".yaml"):
                    # load the component from the yaml
                    path = os.path.join(root, file)
                    cpnt = self.load_component_from_spec_yaml(path)
                    if cpnt: # valid component
                        name_to_componnets[cpnt.name] = cpnt
                        # if cpnt.name not in name_to_componnets:
                        #     name_to_componnets[cpnt.name] = []
                        # name_to_componnets[cpnt.name].append(cpnt)
        print(f"loaded {len(name_to_componnets)} components")
        return name_to_componnets

    def load_component_from_spec_yaml(self, spec_yaml: str) -> Component:
        """Load the component from the spec yaml

        Args:
            spec_yaml (str): spec yaml

        Returns:
            Component: component
        """
        with open(spec_yaml, 'r') as file:
            spec = yaml.safe_load(file)
            if self._is_valid_component_spec(spec):
                component = Component(
                    name=spec['name'],
                    display_name=spec.get('display_name', spec['name']),
                    description=spec.get('description', ""),
                    component_type=spec['type'],
                    path=spec_yaml,
                    inputs=spec.get('inputs', {}).keys()  # only need inputs names
                )
                return component
            else:
                # print(f"invalid component spec: {spec_yaml}")
                return None


    def find_components(
            self,
            name: str,
            component_type: ComponentType = None,
            top_k: int = 5,
            cut_off: float = 0.0,
        ) -> List[Component]:
        """Find top_k components that match the name

        Args:
            name (str): name of the component
            component_type (str, optional): component type. Defaults to None.
            top_k (int, optional): top k components to return. Defaults to 5.

        Returns:
            List[Component]: list of components
        """
        # filter name_to_componnets by component type
        if component_type:
            name_to_componnets = {k: v for k, v in self.name_to_componnets.items() if v.component_type.startswith(component_type)}
        else:
            name_to_componnets = self.name_to_componnets

        matched_names = difflib.get_close_matches(name, name_to_componnets.keys(), top_k, cut_off)
        # get components from self.name_to_componnets given the matched_names
        matched_components = [self.name_to_componnets[matched_name] for matched_name in matched_names]
        return matched_components

    def find_components_by_codex(
            self,
            description: str,
            component_type: ComponentType = None,
            top_k: int = 5,
    ) -> List[Component]:
        """Given description, find top_k components that match the description

        Args:
            description (str): user description of the component
            component_type (ComponentType, optional): component type. Defaults to None.
            top_k (int, optional): top k components to return. Defaults to 5.

        Returns:
            List[Component]: list of components
        """
        # filter name_to_componnets by component type
        if component_type:
            name_to_componnets = {k: v for k, v in self.name_to_componnets.items() if v.component_type.startswith(component_type)}
        else:
            name_to_componnets = self.name_to_componnets
        matched_names = self.codex_component_finder.generate(description, name_to_componnets, top_k)
        # get components from self.name_to_componnets given the matched_names
        matched_components = [self.name_to_componnets[matched_name] for matched_name in matched_names if matched_name in self.name_to_componnets]
        return matched_components


    def _is_valid_component_spec(self, spec: dict) -> bool:
        """Check if the spec is a valid component spec

        Args:
            spec (dict): component spec

        Returns:
            bool: True if the spec is valid
        """
        return all([key in spec for key in ['name', '$schema']])