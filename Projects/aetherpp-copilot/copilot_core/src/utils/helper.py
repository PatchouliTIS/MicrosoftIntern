import os
from .key_vault_util import KeyVaultHelper

def get_codex_token():
    # relative path to the workspace config file
    workspace_config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "workspace", "wxtcstrain.json")
    helper = KeyVaultHelper(workspace_config_path=workspace_config_path)
    secret = helper.get_secret("codex-playground-token")
    return secret

def write_files(file_contents, folder_name):
    """write the files to the folder

    Args:
        file_contents (Dict[str, str]): dictionary containing the file names and contents
        folder_name (str): folder to write the files to
    return:
        file_paths: list of file paths
    """
    os.makedirs(folder_name, exist_ok=True)
    file_paths = []
    for file_name, file_content in file_contents.items():
        file_path = os.path.join(folder_name, file_name)
        with open(file_path, 'w') as file:
            file.write(file_content)
        file_paths.append(file_path)
    return file_paths