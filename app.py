from fastai.vision.all import (
    load_learner,
    PILImage,
)
import streamlit as st
import numpy as np
import pickle
import cv2
import time
from streamlit_webrtc import webrtc_streamer
from pathlib import Path
import torch
import urllib.request
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import mediapipe as mp

#make detection mediapipe
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

#python -m streamlit run app.py
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)
cap.set(3,224)
cap.set(4,224)
output_placeholder = st.empty()
pred_placeholder = st.empty()
MODEL_URL = "https://huggingface.co/spaces/farrr/Sitting-Poseture-Estimate/resolve/main/Sitting-Poseture-Estimate-model-2-final.pkl"
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
    '''if mp_model.pose_landmarks:
        mpDraw.draw_landmarks(frame2, mp_model.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(mp_model.pose_landmarks.landmark):
            h ,w ,c  = frame2.shape
            print(id,lm)
            cx ,cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame2, (cx,cy),4, (255, 0, 0), cv2.FILLED)'''
    FRAME_WINDOW.image(frame2)

def main():
    st.title('Sitting Poseture Estimate')
    st.markdown("<hr>", unsafe_allow_html=True)

    while True:
        frame()
        result = predict(learn_inf, frame2[1])
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