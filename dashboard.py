# Python In-built packages
from pathlib import Path

# External packages
import streamlit as st

# Local Modules
import settings
import helper

def main():
    # Setting page layout
    st.set_page_config(
        page_title="Human Detection using YOLOv8",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main page heading
    st.title("Human Detection as Automation Indicator")
    
    # Sidebar
    st.sidebar.header("ML Model Detection Config")
    
    # Model Options
    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 25, 100, 40)) / 100
    
    # Selecting Detection Or Segmentation
    model_path = Path(settings.DETECTION_MODEL)
    
    
    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
    
    st.sidebar.header("Image/Video Config")
    source_radio = st.sidebar.radio(
        "Select Source", settings.SOURCES_LIST)
    
    
    # If image is selected
    if source_radio == settings.WEBCAM:
        helper.play_webcam(confidence, model)
    
    elif source_radio == settings.RTSP:
        helper.play_rtsp_stream(confidence, model)
    
    else:
        st.error("Please select a valid source type!")
    
if __name__ == '__main__':
    main()
