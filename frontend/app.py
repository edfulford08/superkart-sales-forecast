import os

import pandas as pd
import requests
import streamlit as st

# Newer Streamlit versions replace use_container_width=True with width="stretch"; use whichever the installed version understands
try:
    _streamlit_version = tuple(int(part) for part in st.__version__.split(".")[:2])
except ValueError:
    _streamlit_version = (0, 0)
STRETCH = {"width": "stretch"} if _streamlit_version >= (1, 50) else {"use_container_width": True}

# Inside the Docker network the backend container is reachable by its container name "backend".
# Set BACKEND_URL to use another address (for example the forwarded URL) and API_KEY if the backend requires one.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:7860").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

FALLBACK_CHOICES = {
    "Product_Type": ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods", "Fruits and Vegetables", "Hard Drinks",
                     "Health and Hygiene", "Household", "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"],
    "Product_Sugar_Content": ["Low Sugar", "No Sugar", "Regular"],
    "Store_Size": ["High", "Medium", "Small"],
    "Store_Location_City_Type": ["Tier 1", "Tier 2", "Tier 3"],
    "Store_Type": ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"],
}
TEMPLATE_COLUMNS = ["Product_Id", "Store_Id", "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_Type",
                    "Product_MRP", "Store_Establishment_Year", "Store_Size", "Store_Location_City_Type", "Store_Type"]


def call_backend(method, path, **kwargs):
    """Call the backend and return (response or None, short problem text)."""
    try:
        return requests.request(method, BACKEND_URL + path, headers=HEADERS, timeout=kwargs.pop("timeout", 30), **kwargs), ""
    except requests.exceptions.RequestException as err:
        return None, type(err).__name__


def read_error(response):
    try:
        return str(response.json().get("error", response.text))
    except ValueError:
        return response.text[:300]


@st.cache_data(ttl=300, show_spinner=False)
def load_schema(url, key):
    """Allowed values and defaults come from the backend (read from the trained model) with a built-in fallback."""
    try:
        response = requests.get(url + "/v1/schema", headers={"X-API-Key": key} if key else {}, timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


def choices(schema, column):
    if schema and column in schema.get("categories", {}):
        return schema["categories"][column]
    return FALLBACK_CHOICES[column]


def number_limits(schema, column, low, high, default):
    if schema and column in schema.get("numeric", {}):
        rule = schema["numeric"][column]
        return float(rule["min"]), float(rule["max"]), float(schema["defaults"].get(column, default))
    return low, high, default


st.set_page_config(page_title="SuperKart Sales Forecaster", layout="wide")
st.title("SuperKart Product-Store Sales Forecast")
st.write("Estimate the total sales revenue of a product in a given store, with a forecast range. Use the first tab for one product, the second tab to score a CSV file, and the third tab for model information.")

health, reason = call_backend("GET", "/", timeout=10)
ready = health is not None and health.status_code == 200
if not ready:
    st.warning("The forecast service (" + BACKEND_URL + ") is not reachable yet" + ((": " + reason) if reason else "") + ". Check that the backend container is running and reload the page.")
schema = load_schema(BACKEND_URL, API_KEY) if ready else None

tab_single, tab_batch, tab_info = st.tabs(["Single forecast", "Batch forecast (CSV)", "Model information"])

with tab_single:
    types = choices(schema, "Product_Type")
    with st.form("single_forecast"):
        left, right = st.columns(2)
        with left:
            st.subheader("Product")
            product_id = st.text_input("Product Id (two letters followed by digits)", "FD6114")
            product_type = st.selectbox("Product type", types, index=types.index("Fruits and Vegetables") if "Fruits and Vegetables" in types else 0)
            sugar_options = choices(schema, "Product_Sugar_Content")
            sugar = st.selectbox("Sugar content", sugar_options)
            w_low, w_high, w_def = number_limits(schema, "Product_Weight", 1.0, 40.0, 12.7)
            weight = st.number_input("Product weight", min_value=w_low, max_value=w_high, value=min(max(12.7, w_low), w_high), step=0.1)
            m_low, m_high, m_def = number_limits(schema, "Product_MRP", 1.0, 500.0, 147.0)
            mrp = st.number_input("Product MRP", min_value=m_low, max_value=m_high, value=min(max(147.0, m_low), m_high), step=0.5)
            a_low, a_high, a_def = number_limits(schema, "Product_Allocated_Area", 0.001, 1.0, 0.069)
            area = st.number_input("Allocated display area ratio", min_value=max(a_low, 0.0), max_value=a_high, value=min(max(0.069, a_low), a_high), step=0.001, format="%.3f")
        with right:
            st.subheader("Store")
            store_types = choices(schema, "Store_Type")
            store_type = st.selectbox("Store type", store_types, index=store_types.index("Supermarket Type2") if "Supermarket Type2" in store_types else 0)
            sizes = choices(schema, "Store_Size")
            store_size = st.selectbox("Store size", sizes, index=sizes.index("Medium") if "Medium" in sizes else 0)
            cities = choices(schema, "Store_Location_City_Type")
            city_type = st.selectbox("City tier", cities, index=cities.index("Tier 2") if "Tier 2" in cities else 0)
            year = st.number_input("Store establishment year", min_value=1950, max_value=2026, value=2009, step=1)
        submitted = st.form_submit_button("Forecast sales")

    if submitted:
        payload = {
            "Product_Id": product_id, "Product_Weight": weight, "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": area, "Product_Type": product_type, "Product_MRP": mrp,
            "Store_Establishment_Year": int(year), "Store_Size": store_size,
            "Store_Location_City_Type": city_type, "Store_Type": store_type,
        }
        response, problem = call_backend("POST", "/v1/forecast", json=payload, timeout=90)
        if response is None:
            st.error("Could not reach the backend: " + problem)
        elif response.status_code == 200:
            try:
                body = response.json()
                st.metric("Predicted total sales", "{:,.2f}".format(body["Predicted_Product_Store_Sales_Total"]))
                if "Lower_Bound" in body:
                    st.write("Forecast range: **{:,.2f}** to **{:,.2f}** (use the range, not the single number, to set safety stock).".format(body["Lower_Bound"], body["Upper_Bound"]))
            except (KeyError, ValueError) as err:
                st.error("The backend returned an unexpected response: " + str(err))
        else:
            st.error("The API returned an error: " + read_error(response))

with tab_batch:
    st.write("Upload a CSV with the columns: " + ", ".join(TEMPLATE_COLUMNS[:1] + TEMPLATE_COLUMNS[2:]) + ". Product_Id and Store_Id (if present) are returned with each prediction.")
    template = pd.DataFrame([{"Product_Id": "FD6114", "Store_Id": "OUT004", "Product_Weight": 12.66, "Product_Sugar_Content": "Low Sugar",
                              "Product_Allocated_Area": 0.027, "Product_Type": "Frozen Foods", "Product_MRP": 117.08,
                              "Store_Establishment_Year": 2009, "Store_Size": "Medium", "Store_Location_City_Type": "Tier 2",
                              "Store_Type": "Supermarket Type2"}], columns=TEMPLATE_COLUMNS)
    st.download_button("Download a template CSV", template.to_csv(index=False), "superkart_template.csv", "text/csv")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None and st.button("Forecast batch"):
        response, problem = call_backend("POST", "/v1/forecastbatch", files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")}, timeout=180)
        if response is None:
            st.error("Could not reach the backend: " + problem)
        elif response.status_code == 200:
            try:
                result = pd.DataFrame(response.json()["predictions"])
                st.success("Predictions completed successfully for " + str(len(result)) + " rows.")
                st.dataframe(result, **STRETCH)
                st.download_button("Download predictions", result.to_csv(index=False), "superkart_forecasts.csv", "text/csv")
                if "Store_Id" in result.columns:
                    st.subheader("Forecast total by store")
                    rollup = result.groupby("Store_Id")["Predicted_Product_Store_Sales_Total"].sum()
                    st.bar_chart(rollup)
                    st.dataframe(rollup.round(2).rename("Forecast total").to_frame(), **STRETCH)
            except (KeyError, ValueError) as err:
                st.error("The backend returned an unexpected response: " + str(err))
        else:
            st.error("The API returned an error: " + read_error(response))

with tab_info:
    response, problem = call_backend("GET", "/v1/metadata", timeout=15)
    if response is None:
        st.info("Model information is not available: " + problem)
    elif response.status_code == 200:
        info = response.json()
        st.write("Model version: **" + str(info.get("model_version")) + "** (" + str(info.get("model_name")) + ")")
        if "metrics" in info:
            st.dataframe(pd.Series(info["metrics"]).rename("Value").to_frame(), **STRETCH)
        if "interval" in info:
            st.write("Forecast range: " + str(info["interval"].get("coverage", "")) + " coverage, estimated by split conformal prediction per store type.")
    else:
        st.info("Model information is not available: " + read_error(response))
