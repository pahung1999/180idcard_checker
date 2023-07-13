from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import requests
from io import BytesIO
import time


url = "http://10.10.10.92:8000/crop/img"
weight_path = "./weight/4labels/mobilenet_v3_small_best_acc_w224_h224.pth"


# Load the pre-trained MobileNetV3 model
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1, progress = True)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 4)

# Load the pre-trained weights
model.load_state_dict(torch.load(weight_path))
model.eval()

# Preprocess the result image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Streamlit page title
st.title("Image Segmentation")

# Display input image
input_image = st.file_uploader("Upload Input Image", type=["jpg", "jpeg"])
if input_image is not None:
    st.image(input_image, caption="Input Image", use_column_width=True)

    # Checkbox to perform rotation
    rotate_check = st.checkbox("Rotation")

    # Button to process the image
    if st.button("Process Image"):
        
        post_param = dict(auto_rotate=False)

        # Convert input image to bytes
        img_bytes = input_image.read()

        a1 = time.time()
        # Send request to the server
        res = requests.post(url, files=dict(image=img_bytes), data=post_param)
        b1 = time.time()

        # Convert result image bytes to PIL Image
        segment_img = Image.open(BytesIO(res.content))

        if not rotate_check:
            # Display the result image
            st.image(segment_img, caption=f"Segment Image in {b1-a1} s", use_column_width=True)
        else:
            a2 = time.time()
            test_image = transform(segment_img)
            test_image = test_image.unsqueeze(0)  # Add a batch dimension

            # Make predictions on the test image
            
            with torch.no_grad():
                outputs = model(test_image)
                _, predicted = torch.max(outputs, 1)
            
            # Get the predicted label
            predicted_label = 360 * predicted.item() / 4

            # Rotate the result image
            rotated_image = segment_img.rotate(-int(predicted_label))
            b2 = time.time()
            # # Display the rotated image
            # st.image(rotated_image, caption=f"Rotated Image in {b-a} s", use_column_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Segmented Image")
                st.image(segment_img, caption=f"Segment Image in {b1-a1} s", use_column_width=True)
            with col2:
                st.subheader("Rotated Image")
                st.image(rotated_image, caption=f"Rotated Image in {b2-a2} s", use_column_width=True)