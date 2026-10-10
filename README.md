# SuperKart Sales Forecast (Flask API + Streamlit frontend)

Two-container application that forecasts the total sales revenue of a product in a SuperKart store
(`Product_Store_Sales_Total`) with a model trained in the accompanying notebook. Every forecast comes with a
90 percent forecast range.

## Layout

- `backend/` - Flask API served by gunicorn on port 7860 (`app.py`, `superkart_features.py`, serialized model, `model_metadata.json`, `requirements.txt`, `Dockerfile`)
- `frontend/` - Streamlit app on port 8501 (`app.py`, `requirements.txt`, `Dockerfile`)
- `training/` - the wide hyperparameter search script and its cached result (`superkart_search.py`, `search_cache.json`)
- `tests/` - automated tests (`python -m unittest discover -s tests -v`)
- `model_registry.json` - every model version with its status, metrics and file fingerprint
- `run_containers.sh` - builds both images, creates the Docker network, and starts both containers
- `.devcontainer/devcontainer.json` - makes a Codespace install Docker, forward ports 7860 and 8501 and start both containers by itself
- `.github/workflows/ci.yml` - tests, dependency audit and Docker build on every push (uploaded when the token has the `workflow` scope)

## Run in a GitHub Codespace

```
bash run_containers.sh
```

or step by step:

```
cd backend
docker build -t superkart-backend .
cd ../frontend
docker build -t superkart-frontend .
docker network create superkart-app-network
docker run -d --name backend --network superkart-app-network -p 7860:7860 superkart-backend
docker run -d --name frontend --network superkart-app-network -p 8501:8501 superkart-frontend
```

A Codespace created from this repository runs the script automatically (see `.devcontainer/devcontainer.json`; progress is in `run_containers.log`).
Make ports 7860 and 8501 **Public** in the PORTS tab and open the address of port 8501.
Stop the containers with `docker stop backend frontend` and stop the Codespace when you are finished.

## API

- `GET /` health, `GET /v1/schema` allowed values, `GET /v1/metadata` model version and metrics
- `POST /v1/forecast` one record as JSON, `POST /v1/forecastbatch` a CSV file (form field `file`)
- `POST /v1/drift` a CSV of recent inputs, compared with the training data (population stability index)

Set `API_KEY` (clients then send the `X-API-Key` header) and `RATE_LIMIT_PER_MINUTE` as environment variables.
Example: `API_KEY=secret bash run_containers.sh`.

## Monitoring and retraining

Run `/v1/drift` on each month's new records and compare the forecast error with `metrics.baseline_mae_out_of_fold` in
`model_metadata.json`. Retrain when inputs drift significantly (PSI at or above 0.25), when the error exceeds
1.25 times the baseline, or when the model is older than 90 days (the policy is stored in the metadata).
