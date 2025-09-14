# Enhanced Streamlit App for Iris Model Deployment
# Save this code in a file named 'streamlit_app_enhanced.py' and run using: streamlit run streamlit_app_enhanced.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configure the page
st.set_page_config(
    page_title="Iris Flower Prediction - withArijit",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }

    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('iris_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'iris_model.pkl' not found. Please run the training code first.")
        return None

model = load_model()

# App Header
st.markdown('<h1 class="main-header">🌸 Iris Flower Species Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #666;">Powered by <strong>Learn with Arijit</strong> (withArijit.com)</p>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown('<h2 class="sub-header">🔧 Model Parameters</h2>', unsafe_allow_html=True)

    # Information about the features
    with st.expander("ℹ️ About Iris Features", expanded=False):
        st.write("""
        **Sepal Length**: Length of the outer part of the flower (4.0-8.0 cm)

        **Sepal Width**: Width of the outer part of the flower (2.0-4.5 cm)

        **Petal Length**: Length of the inner part of the flower (1.0-7.0 cm)

        **Petal Width**: Width of the inner part of the flower (0.1-2.5 cm)
        """)

    st.markdown("### Input Features")

    # Feature inputs with better styling
    sepal_length = st.slider(
        '🌱 Sepal Length (cm)', 
        min_value=4.0, 
        max_value=8.0, 
        value=5.4, 
        step=0.1,
        help="The length of the sepal in centimeters"
    )

    sepal_width = st.slider(
        '🌱 Sepal Width (cm)', 
        min_value=2.0, 
        max_value=4.5, 
        value=3.4, 
        step=0.1,
        help="The width of the sepal in centimeters"
    )

    petal_length = st.slider(
        '🌺 Petal Length (cm)', 
        min_value=1.0, 
        max_value=7.0, 
        value=1.3, 
        step=0.1,
        help="The length of the petal in centimeters"
    )

    petal_width = st.slider(
        '🌺 Petal Width (cm)', 
        min_value=0.1, 
        max_value=2.5, 
        value=0.2, 
        step=0.1,
        help="The width of the petal in centimeters"
    )

    # Quick preset buttons
    st.markdown("### 🎯 Quick Presets")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Setosa", help="Typical Setosa values", use_container_width=True):
            st.session_state.sepal_length = 5.1
            st.session_state.sepal_width = 3.5
            st.session_state.petal_length = 1.4
            st.session_state.petal_width = 0.2

    with col2:
        if st.button("Versicolor", help="Typical Versicolor values", use_container_width=True):
            st.session_state.sepal_length = 7.0
            st.session_state.sepal_width = 3.2
            st.session_state.petal_length = 4.7
            st.session_state.petal_width = 1.4

    with col3:
        if st.button("Virginica", help="Typical Virginica values", use_container_width=True):
            st.session_state.sepal_length = 6.3
            st.session_state.sepal_width = 3.3
            st.session_state.petal_length = 6.0
            st.session_state.petal_width = 2.5

# Main content area
if model is not None:
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<h2 class="sub-header">📊 Current Input Values</h2>', unsafe_allow_html=True)

        # Display current inputs in a nice format
        input_df = pd.DataFrame({
            'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width],
            'Range': ['4.0 - 8.0', '2.0 - 4.5', '1.0 - 7.0', '0.1 - 2.5']
        })

        st.dataframe(input_df, use_container_width=True, hide_index=True)

        # Visual representation of features
        st.markdown("### 📏 Feature Measurements")

        # Create metrics for each feature
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                label="🌱 Sepal Length", 
                value=f"{sepal_length} cm",
                delta=f"{sepal_length - 5.84:.1f}" if sepal_length != 5.84 else None,
                help="Average is 5.84 cm"
            )
            st.metric(
                label="🌺 Petal Length", 
                value=f"{petal_length} cm",
                delta=f"{petal_length - 3.76:.1f}" if petal_length != 3.76 else None,
                help="Average is 3.76 cm"
            )

        with col_b:
            st.metric(
                label="🌱 Sepal Width", 
                value=f"{sepal_width} cm",
                delta=f"{sepal_width - 3.06:.1f}" if sepal_width != 3.06 else None,
                help="Average is 3.06 cm"
            )
            st.metric(
                label="🌺 Petal Width", 
                value=f"{petal_width} cm",
                delta=f"{petal_width - 1.20:.1f}" if petal_width != 1.20 else None,
                help="Average is 1.20 cm"
            )

    with col2:
        st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)

        # Make prediction
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Species mapping
        species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        species_emojis = {0: '🌸', 1: '🌺', 2: '🌻'}
        predicted_species = species_map[prediction[0]]
        predicted_emoji = species_emojis[prediction[0]]
        confidence = max(prediction_proba[0])

        # Display prediction with styling
        st.markdown(f"""
        <div class="prediction-box">
            <h3>{predicted_emoji} Predicted Species</h3>
            <h2>{predicted_species}</h2>
            <p>Confidence: {confidence:.2%}</p>
        </div>
        """, unsafe_allow_html=True)

        # Show all probabilities
        st.markdown("### 📈 All Probabilities")

        for i, (species, prob) in enumerate(zip(['Setosa 🌸', 'Versicolor 🌺', 'Virginica 🌻'], prediction_proba[0])):
            st.write(f"**{species}**: {prob:.2%}")
            st.progress(prob)

        # Confidence indicator
        if confidence > 0.8:
            st.success("🎯 High Confidence Prediction!")
        elif confidence > 0.6:
            st.warning("⚡ Moderate Confidence")
        else:
            st.error("⚠️ Low Confidence - Check inputs")

    # Additional Information Section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🌺 About Iris Species</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🌸 Iris Setosa</h4>
            <p><strong>Characteristics:</strong><br/>
            • Small flowers<br/>
            • Short, wide petals<br/>
            • Found in Arctic regions<br/>
            • Easy to distinguish</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🌺 Iris Versicolor</h4>
            <p><strong>Characteristics:</strong><br/>
            • Medium-sized flowers<br/>
            • Moderate petal dimensions<br/>
            • Native to North America<br/>
            • Blue flag iris</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>🌻 Iris Virginica</h4>
            <p><strong>Characteristics:</strong><br/>
            • Large flowers<br/>
            • Long, wide petals<br/>
            • Southeastern United States<br/>
            • Most variable species</p>
        </div>
        """, unsafe_allow_html=True)

    # Model Information
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🤖 Model Information</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Model Type:** Random Forest Classifier

        **Features Used:** 4 numerical features
        - Sepal Length & Width
        - Petal Length & Width

        **Classes:** 3 Iris species
        """)

    with col2:
        st.info("""
        **Dataset:** Famous Iris dataset by R.A. Fisher (1936)

        **Training Size:** 150 samples (50 per species)

        **Application:** Botanical classification and ML education
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
        <h4>🎓 Learn with Arijit - AI Training & Certification</h4>
        <p>🌐 Visit <a href="http://witharijit.com" target="_blank" style="color: #1f77b4;">withArijit.com</a> 
        for comprehensive AI & Machine Learning courses</p>
        <p><em>This demonstration showcases ML model deployment using Streamlit, FastAPI, and scikit-learn.<br/>
        Perfect for understanding production-ready AI applications!</em></p>
        <p><strong>📧 Contact:</strong> Learn more about our 4-month AI Certification Program</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("⚠️ Model could not be loaded. Please ensure 'iris_model.pkl' exists in the same directory.")
    st.info("💡 Run the model training code first to generate the required model file.")

    # Instructions for users
    st.markdown("""
    ### 📋 Setup Instructions:

    1. **Run the training code** from your notebook to generate `iris_model.pkl`
    2. **Save this Streamlit code** as `streamlit_app_enhanced.py` 
    3. **Install required packages**: `pip install streamlit pandas numpy`
    4. **Run the app**: `streamlit run streamlit_app_enhanced.py`
    """)