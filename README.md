# SuperKart Sales Forecast (Flask API + Streamlit frontend)

Two-container application that forecasts the total sales revenue of a product in a SuperKart store
(`Product_Store_Sales_Total`) with a tuned Random Forest pipeline trained in the accompanying notebook.

## Layout

- `backend/` - Flask API served by gunicorn on port 7860 (`app.py`, serialized model, `requirements.txt`, `Dockerfile`)
- `frontend/` - Streamlit app on port 8501 (`app.py`, `requirements.txt`, `Dockerfile`)
- `run_containers.sh` - builds both images, creates the Docker network, and starts both containers

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

Make ports 7860 and 8501 **Public** in the PORTS tab and open the address of port 8501.
Stop the containers with `docker stop backend frontend` and stop the Codespace when you are finished.
