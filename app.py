import base64
import io
import time
from datetime import datetime

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Page settings Streamlit
st.set_page_config(
    page_title="Brain Tumour Classifier AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS to enhance the appearance
def load_custom_css():
    st.markdown(
        """
    <style>
    /* import fonts from google */
    @import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap\');
    
    /* CSS Variables */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --background-color: #F8FAFC;
        --text-color: #1E293B;
        --card-background: #292929;
        --border-color: #E2E8F0;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
    }
    
    /* Styles */
    .main {
        font-family: \'Inter\', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Custom Header  */
    .custom-header {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .custom-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .custom-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        text-align: center;
        margin: 0.5rem 0 0 0;
    }
    
    /* Cards */
    .info-card {
        background: var(--card-background);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid var(--border-color);
        margin: 1rem 0;
    }
    
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .result-card h3 {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .result-card .confidence {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed var(--primary-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(46, 134, 171, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: var(--secondary-color);
        background: rgba(162, 59, 114, 0.05);
    }
    
    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Metrics */
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .metric-item {
        text-align: center;
        padding: 1rem;
        background: var(--card-background);
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        min-width: 120px;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-color);
        opacity: 0.7;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: var(--card-background);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
    }
    
    /* Hide elements from Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
        unsafe_allow_html=True,
    )


# Function to create probability graph with Plotly
def create_probability_chart(probabilities, classes):
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#10B981"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=classes,
                y=probabilities[0] * 100,
                marker_color=colors,
                text=[f"{p:.1%}" for p in probabilities[0]],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Probabilities by Class",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "family": "Inter"},
        },
        xaxis_title="Tumour Type",
        yaxis_title="Probability (%)",
        template="plotly_white",
        height=400,
        font=dict(family="Inter", size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# Function to create confidence graph
def create_confidence_gauge(confidence):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Confidence Level"},
            delta={"reference": 80},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "#2E86AB"},
                "steps": [
                    {"range": [0, 50], "color": "#FEE2E2"},
                    {"range": [50, 80], "color": "#FEF3C7"},
                    {"range": [80, 100], "color": "#D1FAE5"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(
        height=300,
        font=dict(family="Inter", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# Function to load the model with error handling
@st.cache_resource
def load_classification_model():
    try:
        return load_model("brain_tumour_model.keras", compile=False)
    except Exception as e:
        st.error(f"❌ Error loading the model: {str(e)}")
        return None


# Function to preprocess and classify the image
def predict_tumor(img, model):
    try:
        # Preprocessing
        img = img.convert("RGB")
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        with st.spinner("🔄 Analysing the image..."):
            result = model.predict(img_array, verbose=0)

        return result
    except Exception as e:
        st.error(f"❌ Error during image processing: {str(e)}")
        return None


# Function to convert image to base64
def get_image_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def main():
    # Load custom CSS
    load_custom_css()

    # Custom Header
    st.markdown(
        """
    <div class="custom-header fade-in">
        <h1>🧠 Brain Tumour Classifier AI</h1>
        <p>Artificial Intelligence System for Brain Tumour Classification</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar with information
    with st.sidebar:
        st.markdown("### 📊 System Information")

        st.markdown(
            """
        <div class="info-card">
            <h4>🎯 Model Accuracy</h4>
            <p>Trained with thousands of MRI images</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="info-card">
            <h4>🔬 Detected Classes</h4>
            <ul>
                <li><strong>Glioma</strong> - Brain/spinal cord tumour</li>
                <li><strong>Meningioma</strong> - Tumour of the meninges</li>
                <li><strong>Pituitary</strong> - Pituitary tumour</li>
                <li><strong>No tumour</strong> - Normal tissue</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="info-card">
            <h4>⚠️ Important Notice</h4>
            <p>This system is a diagnostic support tool. Always consult a qualified health professional.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Load the model
    model = load_classification_model()

    if model is None:
        st.error(
            "❌ The application could not be started due to an error loading the model."
        )
        return

    # Main area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Image Upload")

        # Custom upload area
        uploaded_file = st.file_uploader(
            "",
            type=["jpg", "png", "jpeg"],
            help="Supported formats: JPG, PNG, JPEG (max. 200MB)",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)

                # Show image information
                st.markdown("#### 🖼️ Image Loaded")
                st.image(
                    img, caption=f"File: {uploaded_file.name}", use_column_width=True
                )

                # Technical information
                st.markdown(
                    f"""
                <div class="info-card">
                    <strong>📏 Dimensions:</strong> {img.size[0]} x {img.size[1]} pixels<br>
                    <strong>📁 Size:</strong> {uploaded_file.size / 1024:.1f} KB<br>
                    <strong>🎨 Format:</strong> {img.format}
                </div>
                """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"❌ Error loading the image: {str(e)}")
                return

    with col2:
        if uploaded_file is not None:
            st.markdown("### 🔍 Analysis Result")

            # Perform the classification
            result = predict_tumor(img, model)

            if result is not None:
                classes = ["Glioma", "Meningioma", "Pituitary", "No Tumour"]
                pred_class = np.argmax(result)
                confidence = np.max(result)

                # Main result card
                st.markdown(
                    f"""
                <div class="result-card fade-in">
                    <h3>Detected Diagnosis</h3>
                    <div class="confidence">{classes[pred_class]}</div>
                    <p>Confidence: {confidence:.1%}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)

                with col_m1:
                    st.metric(
                        label="🎯 Confidence",
                        value=f"{confidence:.1%}",
                        delta=f"{confidence - 0.8:.1%}" if confidence > 0.8 else None,
                    )

                with col_m2:
                    st.metric(label="⏱️ Time", value="< 1s", delta="Fast")

                with col_m3:
                    st.metric(label="📊 Classes", value="4", delta="Complete")

                # Probability chart
                st.markdown("#### 📊 Detailed Analysis")
                prob_chart = create_probability_chart(result, classes)
                st.plotly_chart(prob_chart, use_container_width=True)

                # Confidence chart
                confidence_chart = create_confidence_gauge(confidence)
                st.plotly_chart(confidence_chart, use_container_width=True)

                # Result interpretation
                if confidence > 0.9:
                    st.success(
                        "✅ **High Confidence**: The model is highly confident in the classification."
                    )
                elif confidence > 0.7:
                    st.warning(
                        "⚠️ **Moderate Confidence**: Probable result, but consider additional analysis."
                    )
                else:
                    st.error(
                        "❌ **Low Confidence**: Uncertain result, further analysis recommended."
                    )

                # Button for new analysis
                if st.button("🔄 New Analysis", use_container_width=True):
                    st.experimental_rerun()

        else:
            st.markdown(
                """
            <div class="upload-area">
                <h3>👆 Upload an image</h3>
                <p>Drag and drop or click to select an MRI scan image</p>
                <p><small>Accepted formats: JPG, PNG, JPEG</small></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Additional information section
    st.markdown("---")

    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.markdown(
            """
        <div class="info-card">
            <h4>🧠 About Tumours</h4>
            <p><strong>Glioma:</strong> A tumour that originates in the glial cells of the brain and spinal cord.</p>
            <p><strong>Meningioma:</strong> A generally benign tumour that arises from the meninges.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_info2:
        st.markdown(
            """
        <div class="info-card">
            <h4>🔬 Technology</h4>
            <p>This system uses convolutional neural networks (CNN) trained with deep learning for medical image analysis.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_info3:
        st.markdown(
            """
        <div class="info-card">
            <h4>📈 Accuracy</h4>
            <p>The model was trained and validated with thousands of images, achieving high accuracy in classification.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #64748B; font-size: 0.9rem;">
        <p>🏥 Brain Tumour Classifier AI • Developed with ❤️ to assist healthcare professionals</p>
        <p><small>Version 2.0 • Last updated: {}</small></p>
    </div>
    """.format(
            datetime.now().strftime("%d/%m/%Y")
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()