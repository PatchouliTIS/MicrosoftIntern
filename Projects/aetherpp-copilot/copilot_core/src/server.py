import os
from dotenv import load_dotenv

from fastapi import FastAPI
from codex_component_creator import CodeXComponentCreator
from component import ComponentType
from component_manager import ComponentManager
from codex_reference_generator import CodexReferenceGenerator


load_dotenv()
COMPONENT_FOLDER = os.getenv("COMPONENT_FOLDER")
print("Start initializing the server...")
component_creator_prompt_path = os.path.join(
    os.path.dirname(__file__), "..", "prompt_templates", "component_creator.txt"
)
COMPONENT_CREATOR = CodeXComponentCreator(component_creator_prompt_path)
COMPONENT_MANAGER = ComponentManager(component_folder=COMPONENT_FOLDER)

ref_gen_prompt_path = os.path.join(
    os.path.dirname(__file__), "..", "prompt_templates", "reference_generator.txt"
)
REFERENCE_GENERATOR = CodexReferenceGenerator(ref_gen_prompt_path)
print("Server initialized")


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/create_component/")
def create_component(description: str, component_type: str = None, path: str = "."):
    generated_files = COMPONENT_CREATOR.generate(description, path=path)
    return {"generated_files": generated_files}

@app.get("/find_component/")
def find_component(name: str, component_type: ComponentType = None, top_k: int = 5, cut_off: float = 0.0):
    # find top_k components that match the name
    retrived_components = COMPONENT_MANAGER.find_components(name, component_type, top_k, cut_off)
    return retrived_components

@app.get("/find_component_by_codex/")
def find_component_by_codex(description: str, component_type: ComponentType = None, top_k: int = 5):
    # find top_k components by codex
    retrived_components = COMPONENT_MANAGER.find_components_by_codex(description, component_type, top_k)
    return retrived_components

@app.get("/generate_reference/")
def generate_reference(context: str, component_name: str):
    if component_name not in COMPONENT_MANAGER.name_to_componnets:
        return {"error": f"Component {component_name} not found"}
    component = COMPONENT_MANAGER.name_to_componnets[component_name]
    reference = REFERENCE_GENERATOR.generate(context, component)
    return reference


@app.get("/new_pipeline/")
def new_pipeline():
    pipeline_skeleton = 'from azure.ml.component import Pipeline, dsl\nfrom smile.utils.pipeline_loader import PipelineLoader\n\n\nconfig = PipelineLoader.get_pipeline_config()\n\n@dsl.pipeline(\n    name="pipeline_name",\n    description="Pipeline description",\n    default_datastore=config.storage.default_datastore,\n    non_pipeline_parameters=[],\n)\ndef pipeline_name() -> Pipeline:\n    '
    return pipeline_skeleton
