from fastai.vision.all import (
    load_learner,
)
import streamlit as st
import numpy as np
import cv2
import urllib.request
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import mediapipe as mp
import time

#make detection mediapipe
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
pose = mpPose.Pose()

#set template
st.title('Sitting Poseture Estimate')
st.markdown("<hr>", unsafe_allow_html=True)
capture=st.empty()
FRAME_WINDOW2 = st.image([])
FRAME_WINDOW = st.image([])
output_placeholder = st.empty()
game_placeholder = st.empty()
textgame_placeholder = st.empty()
time_placeholder = st.empty()
image_placeholder = st.image([])
reset_placeholder = st.empty()




#set camera
num_cam=0

MODEL_URL = "https://huggingface.co/spaces/farrr/Sitting-Poseture-Estimate/resolve/main/Sitting-Poseture-Estimate-model-6-final.pkl"
urllib.request.urlretrieve(MODEL_URL, "Sitting-Poseture-Estimate-model-1-final.pkl")
learn_inf = load_learner('Sitting-Poseture-Estimate-model-1-final.pkl')

def predict(learn, img):
    pred, pred_idx, pred_prob = learn.predict(img)
    if pred=='00':
        return "00",pred_prob[pred_idx]*100
    elif pred=='01':
        return "01",pred_prob[pred_idx]*100
    elif pred=='02':
        return "02",pred_prob[pred_idx]*100

def cam_button():
    global num_cam
    st.sidebar.title('Switch camera')
    num_cam = st.sidebar.radio("Select camera", ["Swicth to Camera 1", "Swicth to Camera 2"])
    if num_cam=='Swicth to Camera 1':
       num_cam=0
    else:
       num_cam=1  
    
        

def segment():
    global output_image, mp_model, bg_image, num_cam,result
    # For static images:
    IMAGE_FILES = []
    BG_COLOR = (192, 192, 192) # gray
    MASK_COLOR = (255, 255, 255) # white
    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0) as selfie_segmentation:
      for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        # Generate solid color images for showing the output selfie segmentation mask.
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        output_image = np.where(condition, fg_image, bg_image)
        cv2.imwrite('/tmp/selfie_segmentation_output' + str(idx) + '.png', output_image)

    # For webcam input:
    BG_COLOR = (192, 192, 192) # gray
    cap = cv2.VideoCapture(num_cam)
    cap.set(3,224)
    cap.set(4,224)
    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1) as selfie_segmentation:
      bg_image = None
      while cap.isOpened():
        success, image = cap.read()
        if not success:
            st.write("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        FRAME_WINDOW2.image(image)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = selfie_segmentation.process(image)
        

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack(
        (results.segmentation_mask,) * 3, axis=-1) > 0.1
        # The background can be customized.
        #   a) Load an image (with the same width and height of the input image) to
        #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
        #   b) Blur the input image by applying image filtering, e.g.,
        #      bg_image = cv2.GaussianBlur(image,(55,55),0)
        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            
        output_image = np.where(condition, image, bg_image)

        mp_model = pose.process(output_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(output_image)
        result = predict(learn_inf, output_image)
        tolerance = 0.01

        if abs(result[1] - 79.0080) < tolerance:
            output_placeholder.error("Please stay on camera.")
        else:
            if result[0] == "00":
                output_placeholder.warning(f"Bad sitting pose with the probability of {result[1]:.02f}%")
            if result[0] == "01":
                output_placeholder.success(f"Good sitting pose with the probability of {result[1]:.02f}%")
            if result[0] == "02":
                output_placeholder.warning(f"Text neck with the probability of {result[1]:.02f}%")
 
def main():
    cam_button()
    segment()
        

if __name__ == '__main__':
    main()

#https://colab.research.google.com/github/kevinash/awesome-ai/blob/main/notebooks/6_PosesAndAction/Pose_MediaPipe.ipynb#scrollTo=nW2TjFyhLvVH
#https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python
#https://huggingface.co/spaces/farrr/Sitting-Poseture-Estimate/tree/main
#https://github.com/FarAritat/Sitting-pose-estimate
#https://app.roboflow.com/fararitat/sitting-poseture-estimates/images/NwICAPJ81ro6UQwrUbfU?jobStatus=assigned&annotationJob=Wq7AFPCutXxhfw6wnvly

#python -m streamlit run app.py