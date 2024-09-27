from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
model_path = r'C:\Users\wealt\Documents\machine learning\Car_Price_Prediction\Models\car_price_prediction_bagging_model.pkl'
label_encoders_path = r'C:\Users\wealt\Documents\machine learning\Car_Price_Prediction\Models\label_encoders.pkl'
scalers_path = r'C:\Users\wealt\Documents\machine learning\Car_Price_Prediction\Models\scalers.pkl'
data = pd.read_csv(
    r'C:\Users\wealt\Documents\machine learning\Car_Price_Prediction\Car_Price_files\Prediction_Dataset\car_dekho_cleaned_dataset.csv',
    low_memory=False
)
image_directory = r'C:\Users\wealt\Documents\machine learning\Car_Models'

# Load the trained model and preprocessing steps
model = joblib.load(model_path, mmap_mode='r')
label_encoders = joblib.load(label_encoders_path)
scalers = joblib.load(scalers_path)

# Set pandas option to handle future downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

# Features used for training
features = ['ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName', 'City', 'mileage', 'Seats', 'car_age', 'brand_popularity', 'mileage_normalized']

# Function to filter data based on user selections
def filter_data(oem=None, model=None, body_type=None, fuel_type=None, seats=None):
    filtered_data = data.copy()
    if oem:
        filtered_data = filtered_data[filtered_data['oem'] == oem]
    if model:
        filtered_data = filtered_data[filtered_data['model'] == model]
    if body_type:
        filtered_data = filtered_data[filtered_data['bt'] == body_type]
    if fuel_type:
        filtered_data = filtered_data[filtered_data['ft'] == fuel_type]
    if seats:
        filtered_data = filtered_data[filtered_data['Seats'] == seats]
    return filtered_data

# Preprocessing function for user input (label encoding applied here only)
def preprocess_input(df):
    df['car_age'] = 2024 - df['modelYear']
    brand_popularity = data.groupby('oem')['price'].mean().to_dict()
    df['brand_popularity'] = df['oem'].map(brand_popularity)
    df['mileage_normalized'] = df['mileage'] / df['car_age']

    # Apply label encoding with error handling for unseen labels
    for column in ['ft', 'bt', 'transmission', 'oem', 'model', 'variantName', 'City']:
        if column in df.columns and column in label_encoders:
            df[column] = df[column].apply(lambda x: label_encoders[column].transform([x])[0] if x in label_encoders[column].classes_ else -1)

    # Apply min-max scaling
    for column in ['km', 'ownerNo', 'modelYear']:
        if column in df.columns and column in scalers:
            df[column] = scalers[column].transform(df[[column]])

    return df

# Function to get image path based on the car model name
def get_car_image_path(car_model_name):
    image_filename = f"{car_model_name}.jpg"
    image_path = os.path.join(image_directory, image_filename)
    if os.path.exists(image_path):
        return image_path
    else:
        return None

# Streamlit Application
st.title("Car Price Prediction")

# Sidebar for user inputs
st.sidebar.header('Input Car Features')

# Set background colors
input_background_color = "lightcoral"  # Light maroon color
result_background_color = "#FFF8E7"  # Cosmic latte or beige color

st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container {{
        background-color: {result_background_color};
    }}
    .stButton>button {{
        background-color: lightblue;
        color: white;
    }}
    .result-container {{
        text-align: center;
        background-color: {result_background_color};
        padding: 20px;
        border-radius: 10px;
    }}
    .prediction-title {{
        font-size: 28px;
        color: maroon;
    }}
    .prediction-value {{
        font-size: 36px;
        font-weight: bold;
        color: maroon;
    }}
    .info {{
        font-size: 18px;
        color: grey;
    }}
    .sidebar .sidebar-content {{
        background-color: {input_background_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Get user inputs in a defined order
def visual_selectbox(label, options, index=0):
    selected_option = st.sidebar.selectbox(label, options, index=index)
    return selected_option

# ** Displaying original values for selection **
selected_oem = visual_selectbox('1. Original Equipment Manufacturer (OEM)', sorted(data['oem'].unique()))
filtered_data = filter_data(oem=selected_oem)
selected_model = visual_selectbox('2. Car Model', sorted(filtered_data['model'].unique()))
filtered_data = filter_data(oem=selected_oem, model=selected_model)
body_type = visual_selectbox('3. Body Type', sorted(filtered_data['bt'].unique()))
filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type)
fuel_type = visual_selectbox('4. Fuel Type', sorted(filtered_data['ft'].unique()))
filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type, fuel_type=fuel_type)
transmission = visual_selectbox('5. Transmission Type', sorted(filtered_data['transmission'].unique()))
seat_count = visual_selectbox('6. Seats', sorted(filtered_data['Seats'].unique()))
filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type, fuel_type=fuel_type, seats=seat_count)
selected_variant = visual_selectbox('7. Variant Name', sorted(filtered_data['variantName'].unique()))
modelYear = st.sidebar.number_input('8. Year of Manufacture', min_value=1980, max_value=2024, value=2015)
ownerNo = st.sidebar.number_input('9. Number of Previous Owners', min_value=0, max_value=10, value=1)
km = st.sidebar.number_input('10. Kilometers Driven', min_value=0, max_value=500000, value=10000)
min_mileage = np.floor(filtered_data['mileage'].min())
max_mileage = np.ceil(filtered_data['mileage'].max())
mileage = st.sidebar.slider('11. Mileage (kmpl)', min_value=float(min_mileage), max_value=float(max_mileage), value=float(min_mileage), step=0.5)
city = visual_selectbox('12. City', sorted(data['City'].unique()))

# Create a DataFrame for user input (before encoding)
user_input_data = {
    'ft': [fuel_type],
    'bt': [body_type],
    'km': [km],
    'transmission': [transmission],
    'ownerNo': [ownerNo],
    'oem': [selected_oem],
    'model': [selected_model],
    'modelYear': [modelYear],
    'variantName': [selected_variant],
    'City': [city],
    'mileage': [mileage],
    'Seats': [seat_count],
    'car_age': [2024 - modelYear],
    'brand_popularity': [data.groupby('oem')['price'].mean().to_dict().get(selected_oem)],
    'mileage_normalized': [mileage / (2024 - modelYear)]
}
user_df = pd.DataFrame(user_input_data)

# **Preprocess user input data** (apply encoding/scaling here before prediction)
user_df = preprocess_input(user_df)

# Button to trigger prediction
if st.sidebar.button('Predict'):
    if user_df.notnull().all().all():
        try:
            # Make prediction
            prediction = model.predict(user_df)
            
            # Get the image path based on the selected car model
            car_image_path = get_car_image_path(selected_model)
            
            if car_image_path:
                st.image(car_image_path, caption=f"{selected_model}", use_column_width=True)
            else:
                st.write("Image not found for the selected car model.")
            
            # Display prediction result
            st.markdown(f"""
                <div class="result-container">
                    <h2 class="prediction-title">Predicted Car Price</h2>
                    <p class="prediction-value">₹{prediction[0]:,.2f}</p>
                    <p class="info">Car Age: {user_df['car_age'][0]} years</p>
                    <p class="info">Efficiency Score: {user_df['mileage_normalized'][0]:,.2f} km/year</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Fix: Calculate `car_age` for the data DataFrame
            data['car_age'] = 2024 - data['modelYear']
            
            # Graph Overview: Car Age vs Price
            fig, ax = plt.subplots()
            sns.scatterplot(data=data, x='car_age', y='price', ax=ax)
            ax.set_title('Car Age vs Price Overview')
            ax.set_xlabel('Car Age (Years)')
            ax.set_ylabel('Price (₹)')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
        missing_fields = [col for col in user_df.columns if user_df[col].isnull().any()]
        st.error(f"Missing fields: {', '.join(missing_fields)}. Please fill all required fields.")