from PIL import Image
import streamlit as st
import requests
from io import BytesIO
import time
import os
# IMG_20230706_113750202
def get_image_paths(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                image_paths.append(file_path)
    return image_paths

MAX_SIZE = (1500, 1500)
# 10.10.1.37:10041/
# 10.10.10.92:8000/
url_crop = "http://10.10.10.92:8000/crop/img"
url_mask = "http://10.10.10.92:8000/segmentation-mask"
# url_crop = "http://10.10.1.37:10041/crop/img"
# url_mask = "http://10.10.1.37:10041/segmentation-mask"
post_param = dict(auto_rotate = True)


# image_dir = "/home/phung/AnhHung/temp/hungpa_cccd/wrong/"
image_dir = "/home/phung/AnhHung/temp/hungpa_cccd/hungpa_idcard_for_segment/"



# Get the list of image files in the specified directory
image_files = os.listdir(image_dir)
image_paths = get_image_paths(image_dir)
# Streamlit page title
st.markdown("### Image Segmentation")
st.markdown(f"#### {image_dir}")

# Checkbox to show the origin image
show_origin = st.checkbox("Show Origin")

# Display images in three columns if "Show Origin" checkbox is checked
if show_origin:
    # col1, col2, col3 = st.columns(3)
    # for image_file in image_files:
    for image_path in image_paths:
        # Read the image
        # image_path = os.path.join(image_dir, image_file)
        image_file = os.path.basename(image_path)
        input_image = Image.open(image_path)
        input_image.thumbnail(MAX_SIZE, Image.ANTIALIAS)

        # Convert the input image to bytes
        image_bytes = BytesIO()
        input_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        # Send request for cropping
        crop_res = requests.post(url_crop, files= {'image': image_bytes}, data = post_param)

        # Convert crop response to PIL Image
        crop_img = Image.open(BytesIO(crop_res.content))

        # Send request for segmentation mask
        input_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        mask_res = requests.post(url_mask, files= {'image': image_bytes})

        # Convert mask response to PIL Image
        mask_img = Image.open(BytesIO(mask_res.content))

        # Display the images in three columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Input Image")
            st.image(input_image, caption=image_file, use_column_width=True)
            st.download_button("Download origin", data=open(image_path, "rb").read(), file_name=image_file)

        with col2:
            st.markdown("#### Mask Image")
            st.image(mask_img, caption=image_file, use_column_width=True)

        with col3:
            st.markdown("#### Crop Image")
            st.image(crop_img, caption=image_file, use_column_width=True)

        # Add a separator between rows
        st.markdown("---")

        # Introduce a delay between displaying each row
        # time.sleep(0.1)

# Display images in two columns if "Show Origin" checkbox is not checked
else:
    # col1, col2 = st.columns(2)
    # for image_file in image_files:
    for image_path in image_paths:
        # Read the image
        # image_path = os.path.join(image_dir, image_file)
        image_file = os.path.basename(image_path)
        # Read the image
        input_image = Image.open(image_path)
        input_image.thumbnail(MAX_SIZE)

        # Convert the input image to bytes
        image_bytes = BytesIO()
        input_image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        # Send request for cropping
        crop_res = requests.post(url_crop, files= {'image': image_bytes}, data = post_param)

        # Convert crop response to PIL Image
        crop_img = Image.open(BytesIO(crop_res.content))

        # Send request for segmentation mask
        input_image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        mask_res = requests.post(url_mask, files= {'image': image_bytes})

        # Convert mask response to PIL Image
        mask_img = Image.open(BytesIO(mask_res.content))

        # Display the images in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Mask Image")
            st.image(mask_img, caption=image_file, use_column_width=True)

        with col2:
            st.markdown("#### Crop Image")
            st.image(crop_img, caption=image_file, use_column_width=True)

        # Add a separator between rows
        st.markdown("---")

        # Introduce a delay between displaying each row
        # time.sleep(0.1)
