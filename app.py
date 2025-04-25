import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from facenet_pytorch import MTCNN

# Load a small ResNet18 model (randomly initialized, no external file)
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 1)
model.eval()

# Load MTCNN face detector
@st.cache_resource
def load_detector():
    return MTCNN(keep_all=True)

detector = load_detector()

st.title("WhatsApp Age Estimator (Lite)")

uploaded_file = st.file_uploader("Upload a WhatsApp Profile Picture")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    boxes, probs = detector.detect(img)
    if boxes is not None:
        for box in boxes:
            face = img.crop(box).resize((224, 224))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            face_tensor = transform(face).unsqueeze(0)
            age_estimation = model(face_tensor).item()
            st.write(f"Estimated Age (approx): {abs(int(age_estimation))} years")
    else:
        st.warning("No face detected!")
