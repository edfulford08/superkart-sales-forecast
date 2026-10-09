import os

import pandas as pd
import requests
import streamlit as st

# Inside the Docker network the backend container is reachable by its container name "backend".
# Set the BACKEND_URL environment variable to use another address (for example the forwarded URL).
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:7860").rstrip("/")

PRODUCT_TYPES = ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
                 "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
                 "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"]


def backend_is_ready():
    # Ask the backend health route whether it is up and has loaded the model
    try:
        response = requests.get(BACKEND_URL + "/", timeout=10)
        return response.status_code == 200, ""
    except requests.exceptions.RequestException as err:
        return False, type(err).__name__


def read_error(response):
    # Return the error message of an API response, whether or not it contains JSON
    try:
        return str(response.json().get("error", response.text))
    except ValueError:
        return response.text[:300]


st.set_page_config(page_title="SuperKart Sales Forecaster", layout="wide")
st.title("SuperKart Product-Store Sales Forecast")
st.write("Estimate the total sales revenue of a product in a given store. Use the first tab for one product, or the second tab to score a CSV file of products.")

ready, reason = backend_is_ready()
if not ready:
    st.warning("The forecast service (" + BACKEND_URL + ") is not reachable yet" + ((": " + reason) if reason else "") + ". Check that the backend container is running and reload the page.")

tab_single, tab_batch = st.tabs(["Single forecast", "Batch forecast (CSV)"])

with tab_single:
    with st.form("single_forecast"):
        left, right = st.columns(2)
        with left:
            st.subheader("Product")
            product_id = st.text_input("Product Id (two letters followed by digits)", "FD6114")
            product_type = st.selectbox("Product type", PRODUCT_TYPES, index=PRODUCT_TYPES.index("Fruits and Vegetables"))
            sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
            weight = st.number_input("Product weight", min_value=1.0, max_value=40.0, value=12.7, step=0.1)
            mrp = st.number_input("Product MRP", min_value=1.0, max_value=500.0, value=147.0, step=0.5)
            area = st.number_input("Allocated display area ratio", min_value=0.001, max_value=1.0, value=0.069, step=0.001, format="%.3f")
        with right:
            st.subheader("Store")
            store_type = st.selectbox("Store type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"], index=1)
            store_size = st.selectbox("Store size", ["Small", "Medium", "High"], index=1)
            city_type = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"], index=1)
            year = st.number_input("Store establishment year", min_value=1950, max_value=2026, value=2009, step=1)
        submitted = st.form_submit_button("Forecast sales")

    if submitted:
        payload = {
            "Product_Id": product_id, "Product_Weight": weight, "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": area, "Product_Type": product_type, "Product_MRP": mrp,
            "Store_Establishment_Year": int(year), "Store_Size": store_size,
            "Store_Location_City_Type": city_type, "Store_Type": store_type,
        }
        try:
            response = requests.post(BACKEND_URL + "/v1/forecast", json=payload, timeout=90)
            if response.status_code == 200:
                value = response.json()["Predicted_Product_Store_Sales_Total"]
                st.metric("Predicted total sales", "{:,.2f}".format(value))
            else:
                st.error("The API returned an error: " + read_error(response))
        except requests.exceptions.RequestException as err:
            st.error("Could not reach the backend: " + str(err))
        except (KeyError, ValueError) as err:
            st.error("The backend returned an unexpected response: " + str(err))

with tab_batch:
    st.write("Upload a CSV with the columns: Product_Id, Product_Weight, Product_Sugar_Content, Product_Allocated_Area, Product_Type, Product_MRP, Store_Establishment_Year, Store_Size, Store_Location_City_Type, Store_Type. Product_Id and Store_Id (if present) are returned with each prediction.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None and st.button("Forecast batch"):
        try:
            response = requests.post(BACKEND_URL + "/v1/forecastbatch", files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")}, timeout=180)
            if response.status_code == 200:
                result = pd.DataFrame(response.json()["predictions"])
                st.success("Predictions completed successfully for " + str(len(result)) + " rows.")
                st.dataframe(result, use_container_width=True)
                st.download_button("Download predictions", result.to_csv(index=False), "superkart_forecasts.csv", "text/csv")
            else:
                st.error("The API returned an error: " + read_error(response))
        except requests.exceptions.RequestException as err:
            st.error("Could not reach the backend: " + str(err))
        except (KeyError, ValueError) as err:
            st.error("The backend returned an unexpected response: " + str(err))
