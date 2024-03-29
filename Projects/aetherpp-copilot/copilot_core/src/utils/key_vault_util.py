import os
from azureml.core import Workspace


class KeyVaultHelper:
    """_summary_: This class is used to interact with Azure Key Vault to retrieve secrets."""

    def __init__(self, workspace_config_path: str):
        self.workspace = Workspace.from_config(path=workspace_config_path)
        self.key_vault = self.workspace.get_default_keyvault()

    def get_secret(self, secret_name):
        secret = self.key_vault.get_secret(secret_name)
        return secret


if __name__ == "__main__":
    # relative path to the workspace config file
    workspace_config_path = os.path.join("..", "..", "config", "workspace", "wxtcstrain.json")
    helper = KeyVaultHelper(workspace_config_path=workspace_config_path)
    secret = helper.get_secret("codex-playground-token")
    print(secret)

