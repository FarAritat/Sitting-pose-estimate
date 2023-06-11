from fastai.vision.all import (
    load_learner,
    PILImage,
)
import streamlit as st
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer
from pathlib import Path
import urllib.request
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import mediapipe as mp

#make detection mediapipe
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
pose = mpPose.Pose()

#python -m streamlit run app.py
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)
cap.set(3,224)
cap.set(4,224)
output_placeholder = st.empty()
pred_placeholder = st.empty()
MODEL_URL = "https://huggingface.co/spaces/farrr/Sitting-Poseture-Estimate/resolve/main/Sitting-Poseture-Estimate-model-7-final.pkl"
urllib.request.urlretrieve(MODEL_URL, "Sitting-Poseture-Estimate-model-1-final.pkl")
learn_inf = load_learner('Sitting-Poseture-Estimate-model-1-final.pkl')

def predict(learn, img):
    pred, pred_idx, pred_prob = learn.predict(img)
    if pred=='00':
        return "00",pred_prob[pred_idx]*100
        #st.error(f"Bad sit with the probability of {pred_prob[pred_idx]*100:.02f}%")
    elif pred=='01':
        return "01",pred_prob[pred_idx]*100
        #st.success(f"Good sit with the probability of {pred_prob[pred_idx]*100:.02f}%")
    elif pred=='02':
        return "02",pred_prob[pred_idx]*100
        #st.warning(f"Unknow with the probability of {pred_prob[pred_idx]*100:.02f}%")

def frame():
    global frame2,mp_model
    _, frame = cap.read()
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_model = pose.process(frame2)
    if mp_model.pose_landmarks:
        mpDraw.draw_landmarks(frame2, mp_model.pose_landmarks,mpPose.POSE_CONNECTIONS)
        #for id,lm in enumerate(mp_model.pose_landmarks.landmark):
        #    h ,w ,c  = frame2.shape
        #    print(id,lm)
        #    cx ,cy = int(lm.x * w), int(lm.y * h)
        #    cv2.circle(frame2, (cx,cy),4, (255, 0, 0), cv2.FILLED)
        

def segment():
    global output_image
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
    cap = cv2.VideoCapture(0)
    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1) as selfie_segmentation:
      bg_image = None
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
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
        FRAME_WINDOW.image(output_image)

def main():
    st.title('Sitting Poseture Estimate')
    st.markdown("<hr>", unsafe_allow_html=True)

    while True:
        segment():
        result = predict(learn_inf, output_image)
        if mp_model.pose_landmarks:
            if result[0]=="00":
                output_placeholder.warning(f"Bad sit with the probability of {result[1]:.02f}%")
            if result[0]=="01":
                output_placeholder.success(f"Good sit with the probability of {result[1]:.02f}%")
            if result[0]=="02":
                output_placeholder.warning(f"Text neck with the probability of {result[1]:.02f}%")
        else:
            output_placeholder.error("Please stay on camera   *Tips : Keep your face on the camera*")

if __name__ == '__main__':
    main()

#https://colab.research.google.com/github/kevinash/awesome-ai/blob/main/notebooks/6_PosesAndAction/Pose_MediaPipe.ipynb#scrollTo=nW2TjFyhLvVH
#https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python
#https://huggingface.co/spaces/farrr/Sitting-Poseture-Estimate/tree/main
#https://github.com/FarAritat/Sitting-pose-estimate
#https://app.roboflow.com/fararitat/sitting-poseture-estimates/images/NwICAPJ81ro6UQwrUbfU?jobStatus=assigned&annotationJob=Wq7AFPCutXxhfw6wnvly

#python -m streamlit run app.py