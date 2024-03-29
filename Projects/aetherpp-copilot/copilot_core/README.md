## Conda environment setup

```bash
cd Projects/aetherpp-copilot/copilot_core
conda create -n aetherpp-copilot python=3.9 -y
conda activate aetherpp-copilot
pip install -r requirements.txt
```

## Start FastAPI server

Fill in the `config.env` file with the required environment variables. Then run the following command to start the server.

```bash
cd Projects/aetherpp-copilot/copilot_core/src
uvicorn server:app --env-file config.env [--reload]
```

You could test APIs by visiting `http://127.0.0.1:8000/docs#/` in your browser.

refer to [FastAPI](https://fastapi.tiangolo.com/) for more details.