import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
# import requests # Not needed if batch prediction is handled locally without API call

# Download and load the model
model_path = hf_hub_download(repo_id="drkrishnapatel/SuperkartSalesPrediction", filename="best_superkart_product_sales_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Superkart Sales Prediction
st.title("Superkart Sales Prediction Application")
st.write("""This application predicts sales of each product in different stores based on the product details and the store details where the product is kept.
Please enter the product and store details below to get a prediction.
""")

# User input
Product_Weight = st.number_input("Product_Weight", min_value=0.0, max_value=100.0, step=1.0, value=14.0)
Product_Sugar_Content = st.selectbox("Product_Sugar_Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product_Allocated_Area", min_value=0.0, max_value=0.30, step=0.01, value=0.07)
Product_Type = st.selectbox("Product_Type", ["meat", "snack foods", "hard drinks", "dairy", "canned", "soft drinks", "health and hygiene", "baking goods", "bread", "breakfast", "frozen foods", "fruits and vegetables", "household", "seafood", "starchy foods", "others"])
Product_MRP = st.number_input("Product_MRP", min_value=0.0, max_value=300.0, step=1.0, value=150.0)
Store_Establishment_Year = st.number_input("Store_Establishment_Year", min_value=1980, max_value=2025, step=1, value=1987)
Store_Id = st.selectbox("Store_Id", ["OUT001", "OUT002", "OUT003", "OUT004"])
Store_Size = st.selectbox("Store_Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store_Location_City_Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store_Type", ["Supermarket Type 1", "Supermarket Type 2", "Departmental Store", "Food Mart"])

input_data = pd.DataFrame({
    'Product_Weight': [Product_Weight],
    'Product_Sugar_Content': [Product_Sugar_Content],
    'Product_Allocated_Area': [Product_Allocated_Area],
    'Product_Type': [Product_Type],
    'Product_MRP': [Product_MRP],
    'Store_Id': [Store_Id],
    'Store_Establishment_Year': [Store_Establishment_Year],
    'Store_Size': [Store_Size],
    'Store_Location_City_Type': [Store_Location_City_Type],
    'Store_Type': [Store_Type],
})

# Make prediction when the "Predict" button is clicked
if st.button("Predict Product Sales Total"):
  prediction = model.predict(input_data)[0]
  st.subheader("Prediction for total sales of the product is:")
  st.success(f"The model predicts: **{prediction:.2f}**")


# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch"):
        batch_df = pd.read_csv(uploaded_file)
        batch_predictions = model.predict(batch_df)
        st.success("Batch predictions completed!")
        st.write(pd.DataFrame({'Predicted_Sales': batch_predictions})) # Display the predictions
