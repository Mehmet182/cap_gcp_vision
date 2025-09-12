
import os
import urllib.request

from sdks.novavision.src.base.application import Application
from sdks.novavision.src.base.environment import Environment

app = Application()
env = Environment()


def get_resource_path(config: dict):
    """
    Returns the local file name or downloads the file from storage if needed.
    """
    resource_type = app.get_param(config=config, name="tokenSelection")
    if resource_type == "ConfigPath":
        file_name = app.get_param(config=config, name="storagePath")
        return file_name
    elif resource_type == "ConfigStorage":
        storage_id = app.get_param(config=config, name="storageSource")
        url_path = f"{env.web_api}/storage/default/get-file?id={str(storage_id)}&access-token={env.access_token}"
        local_file_name = f"{storage_id}.json"
        local_file_path = f"/storage/{local_file_name}"
        # Dosya yoksa indir
        if not os.path.exists(local_file_path):
            with urllib.request.urlopen(url_path) as url:
                with open(local_file_path, "wb") as f:
                    f.write(url.read())
        return local_file_name
    else:
        raise ValueError("Unknown resource type!")

def API_AUTH(config: dict):
    resource_type = app.get_param(config=config, name="tokenSelection")
    json_name = get_resource_path(config=config)
    if not json_name.endswith('.json'):
        client_secret_path = f'/storage/{json_name}.json'
    else:
        client_secret_path = f'/storage/{json_name}'
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = client_secret_path
    if not os.path.exists(client_secret_path):
        raise FileNotFoundError(f"{client_secret_path} dosyası bulunamadı")
