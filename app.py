"""Streamlit demo:  streamlit run app.py"""
import pandas as pd
import streamlit as st
from PIL import Image

from data import CLASSES
from inference import Predictor

st.set_page_config(page_title="Driver Distraction Detector", page_icon="🚗", layout="wide")
st.title("🚗 Driver Distraction Detector")
st.caption("Classifies a dashboard-camera frame into 10 driver behaviours "
           "(State Farm Distracted Driver dataset, fine-tuned EfficientNet).")

with st.sidebar:
    ckpt = st.text_input("Model checkpoint", "runs/final/model.pt")
    thr = st.slider("Alert confidence threshold", 0.3, 0.95, 0.6, 0.05)
    st.info("Research demo only. Trained on staged footage from 26 drivers; "
            "not a certified safety system.")


@st.cache_resource
def get_predictor(path):
    return Predictor(path)


try:
    pred = get_predictor(ckpt)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load model at '{ckpt}': {e}")
    st.stop()

tab_up, tab_cam = st.tabs(["Upload image", "Webcam snapshot"])
img = None
with tab_up:
    f = st.file_uploader("Driver image", type=["jpg", "jpeg", "png"])
    if f:
        img = Image.open(f)
with tab_cam:
    c = st.camera_input("Take a photo")
    if c:
        img = Image.open(c)

if img is not None:
    probs = pred.predict(img)
    k, name, conf = pred.describe(probs)
    left, right = st.columns([1, 1])
    left.image(img, use_container_width=True)
    with right:
        if k != 0 and conf >= thr:
            st.error(f"⚠️ DISTRACTED: {name} ({conf:.0%})")
        elif k == 0 and conf >= thr:
            st.success(f"✅ {name} ({conf:.0%})")
        else:
            st.warning(f"Uncertain - best guess: {name} ({conf:.0%})")
        st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASSES))
