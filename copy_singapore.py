import os
import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import gdown

# Page configuration
st.set_page_config(page_title='Singapore Flat Resale Price Prediction')
st.markdown('<h2 style="text-align: center;">Singapore Flat Resale Price Prediction</h2>', unsafe_allow_html=True)

# Google Drive file IDs
model_file_id = '1rbX--6l2p5CCL3cbmTlMtdMywbEXXOUS'
columns_file_id = '1Q1NQoIi6YP5JrQqxxyfhzq4uD4GmFM0H'

# File paths
model_path = os.path.join(os.path.dirname(__file__), 'model_rg_rf.joblib')
columns_path = os.path.join(os.path.dirname(__file__), 'columns_ohe.joblib')

# Function to download files from Google Drive
def download_file_from_google_drive(file_id, destination):
    download_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_url, destination, quiet=False)

# Download the model file if not present
if not os.path.exists(model_path):
    with st.spinner('Downloading the model...'):
        download_file_from_google_drive(model_file_id, model_path)

# Download the columns file if not present
if not os.path.exists(columns_path):
    with st.spinner('Downloading the columns file...'):
        download_file_from_google_drive(columns_file_id, columns_path)

# Load the image
image_path = os.path.join(os.path.dirname(__file__), 'real_estate_image.jpeg')
image = Image.open(image_path)
image = image.resize((700, 200))
st.image(image, use_column_width=False, channels='RGB')

# Load the trained regression model and encoder
model_loaded = False
columns_loaded = False

try:
    # Load the model
    rf_model = load(model_path)
    model_loaded = True
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

try:
    # Load the columns
    X_train_columns = load(columns_path)
    columns_loaded = True
except Exception as e:
    st.error(f"An error occurred while loading the columns: {e}")

# User Input Section
st.write("User Selection")

town = st.selectbox("Town", sorted(['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
       'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
       'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
       'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
       'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
       'TOA PAYOH', 'WOODLANDS', 'YISHUN']))
flat_type = st.selectbox("Flat Type", sorted(['3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', '1 ROOM',
       'MULTI-GENERATION']))
storey_range = st.selectbox("Storey Range", sorted(['07 TO 09', '01 TO 03', '13 TO 15', '10 TO 12', '04 TO 06',
       '19 TO 21', '16 TO 18', '22 TO 24', '25 TO 27', '28 TO 30',
       '34 TO 36', '46 TO 48', '31 TO 33', '37 TO 39', '43 TO 45',
       '40 TO 42', '49 TO 51']))
floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=1, max_value=300)
flat_model = st.selectbox("Flat Model", sorted(['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
       'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
       'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
       'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
       'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen']))
lease_commence_date = st.number_input("Lease Commence Date", min_value=1960, max_value=2023)

# Create a DataFrame with user inputs
user_input = pd.DataFrame({
    'town': [town],
    'flat_type': [flat_type],
    'storey_range': [storey_range],
    'floor_area_sqm': [floor_area_sqm],
    'flat_model': [flat_model],
    'lease_commence_date': [lease_commence_date]
})

# Encode the categorical variables using get_dummies
user_input_encoded = pd.get_dummies(user_input)

# Reindex the encoded DataFrame to match the training columns
if columns_loaded:
    user_input_encoded = user_input_encoded.reindex(columns=X_train_columns, fill_value=0)
else:
    st.error("Prediction cannot proceed as the columns file is not loaded.")

# Make predictions
if st.button("Predict Resale Price"):
    if model_loaded and columns_loaded:
        try:
            predictions = rf_model.predict(user_input_encoded)
            st.success(f"Predicted Resale Price: ${predictions[0]:,.2f}")
        except AttributeError as e:
            st.error(f"AttributeError: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.error("Prediction cannot proceed as the model or the columns file is not loaded.")
