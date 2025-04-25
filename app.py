
import os
import urllib.request

import streamlit as st
import torch
import torchvision.models as models
from PIL import Image
from facenet_pytorch import MTCNN
import cv2

# Automatically download the model if not present
MODEL_PATH = "models/age_model_resnet18_utkface.pth"
MODEL_URL = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/age_model_resnet18_utkface.pth"

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with st.spinner('Downloading model...'):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success('Model downloaded successfully!')

# Load the face detector
@st.cache_resource
def load_model_and_detector():
    mtcnn = MTCNN(keep_all=True)
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return mtcnn, model

mtcnn, model = load_model_and_detector()

# Example Streamlit app
st.title("WhatsApp Age Estimator")

uploaded_file = st.file_uploader("Upload an Image")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    boxes, probs = mtcnn.detect(img)
    if boxes is not None:
        for box in boxes:
            age = model(torch.tensor(img.crop(box).resize((224,224))).permute(2,0,1).unsqueeze(0).float() / 255.0).item()
            st.write(f"Estimated Age: {int(age)}")
    else:
        st.warning("No face detected.")
