import streamlit as st
import numpy as np
import cv2
import pickle

from PIL import Image
from moviepy.editor import VideoFileClip
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from line_fit_video import annotate_image


def detect_lanes(image):
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    img, _, _, _, _ = combined_thresh(undist)
    binary_warped, _, m, m_inv = perspective_transform(img)
    result = annotate_image(binary_warped)
    return result


def process_image(img):
    img = np.array(img)
    result = detect_lanes(img)
    return result


def process_video(video_file):
    video = VideoFileClip(video_file)
    annotated_video = video.fl_image(process_image)
    return annotated_video


# Load camera calibration parameters
with open('calibrate_camera.p', 'rb') as f:
    save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

st.title("Lane Detection App")

option = st.sidebar.selectbox("Choose an option:", ("Image", "Video"))

if option == "Image":
    st.subheader("Lane Detection on Image")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        processed_image, _, _, _, _ = combined_thresh(image)
        result = detect_lanes(processed_image)

        st.image(result, caption="Annotated Image", use_column_width=True)

elif option == "Video":
    st.subheader("Lane Detection on Video")
    video_file = st.file_uploader("Upload a video", type=["mp4"])

    if video_file is not None:
        video = VideoFileClip(video_file)
        st.video(video)
        st.write("Processing video...")
        annotated_video = process_video(video_file)
        st.write("Done! Displaying annotated video...")
        st.video(annotated_video)

