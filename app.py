import streamlit as st
import joblib
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="centered"
)

# Custom CSS
st.markdown(""""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: white;
        border-radius: 20px;
        padding: 20px;
    }
    h1 {
        color: #667eea;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 30px;
        border-radius: 15px;
        margin-top: 20px;
    }
    .price {
        font-size: 3em;
        font-weight: bold;
        color: #667eea;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoder
@st.cache_resource
def load_models():
    try:
        model_path = os.path.join('model', 'house_price_model.pkl')
        encoder_path = os.path.join('model', 'label_encoder.pkl')
        
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        return model, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, label_encoder = load_models()

# Header
st.title("🏠 House Price Prediction System")
st.markdown("### Enter house features to predict the sale price")

if model is None or label_encoder is None:
    st.error("Failed to load model. Please ensure model files exist in the 'model' folder.")
else:
    # Get neighborhoods from encoder
    neighborhoods = list(label_encoder.classes_)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        overall_qual = st.slider(
            "Overall Quality (1-10)",
            min_value=1,
            max_value=10,
            value=7,
            help="Rates the overall material and finish of the house"
        )
        
        gr_liv_area = st.number_input(
            "Living Area (sq ft)",
            min_value=300,
            max_value=10000,
            value=1500,
            step=100,
            help="Above grade (ground) living area square feet"
        )
        
        total_bsmt_sf = st.number_input(
            "Basement Area (sq ft)",
            min_value=0,
            max_value=6000,
            value=1000,
            step=100,
            help="Total square feet of basement area"
        )
    
    with col2:
        garage_cars = st.slider(
            "Garage Size (cars)",
            min_value=0,
            max_value=5,
            value=2,
            help="Size of garage in car capacity"
        )
        
        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2025,
            value=2005,
            step=1,
            help="Original construction date"
        )
        
        neighborhood = st.selectbox(
            "Neighborhood",
            options=neighborhoods,
            help="Physical location within Ames city limits"
        )
    
    # Predict button
    if st.button("🔮 Predict Price", use_container_width=True):
        try:
            # Encode neighborhood
            neighborhood_encoded = label_encoder.transform([neighborhood])[0]
            
            # Prepare features
            features = np.array([[
                overall_qual,
                gr_liv_area,
                total_bsmt_sf,
                garage_cars,
                year_built,
                neighborhood_encoded
            ]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Display result
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### 🎯 Prediction Result")
            st.markdown(f'<div class="price">${prediction:,.2f}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show input summary
            st.markdown("---")
            st.markdown("#### 📋 Input Summary")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write(f"**Overall Quality:** {overall_qual}")
                st.write(f"**Living Area:** {gr_liv_area:,} sq ft")
                st.write(f"**Basement Area:** {total_bsmt_sf:,} sq ft")
            
            with col_b:
                st.write(f"**Garage Cars:** {garage_cars}")
                st.write(f"**Year Built:** {year_built}")
                st.write(f"**Neighborhood:** {neighborhood}")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Powered by Random Forest Regressor | Machine Learning Project</p>",
    unsafe_allow_html=True
)
