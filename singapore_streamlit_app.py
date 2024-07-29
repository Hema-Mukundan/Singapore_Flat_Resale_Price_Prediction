import os
import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import OneHotEncoder
from PIL import Image

# Page configuration
st.set_page_config(page_title='Singapore Flat Resale Price Prediction')
st.markdown('<h2 style="text-align: center;">Singapore Flat Resale Price Prediction</h2>', unsafe_allow_html=True)

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct the path to the image file
image_path = os.path.join(current_dir, 'real_estate_image.jpeg')

# Load and display the image
image = Image.open(image_path)
# Resize the image
desired_width = 700
desired_height = 200
image = image.resize((desired_width, desired_height))

# Display the resized image in Streamlit
st.image(image, use_column_width=False, channels='RGB')

# Load the trained regression model and encoder
model_loaded = False
columns_loaded = False

try:
    # Loading the model
    rf_model = load('C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Singapore_Resale_Flat_Price_Prediction\\model_rg_rf.joblib')
    model_loaded = True
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")

try:
    # Loading the columns
    X_train_columns = load('C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Singapore_Resale_Flat_Price_Prediction\\columns_ohe.joblib')
    columns_loaded = True
except FileNotFoundError:
    st.error("X_train_columns file not found. Please check the file path.")
except Exception as e:
    st.error(f"An unexpected error occurred while loading X_train_columns: {e}")

# User Input Section
st.write("User Selection")

# Example dropdowns, you can replace these with your actual input components
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
    st.error("Prediction cannot proceed as the X_train_columns are not loaded.")

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
        st.error("Prediction cannot proceed as the model or the X_train_columns are not loaded.")
