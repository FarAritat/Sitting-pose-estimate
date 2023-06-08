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
#python -m streamlit run app.py

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
my_placeholder = st.empty()

MODEL_URL = "https://huggingface.co/spaces/farrr/Sitting-Poseture-Estimate/resolve/main/Sitting-Poseture-Estimate-model-1-final.pkl"
urllib.request.urlretrieve(MODEL_URL, "Sitting-Poseture-Estimate-model-1-final.pkl")
learn_inf = load_learner('Sitting-Poseture-Estimate-model-1-final.pkl')

def predict(learn, img):
    pred, pred_idx, pred_prob = learn.predict(img)
    if pred=='00':
        st.success(f"Bad sit with the probability of {pred_prob[pred_idx]*100:.02f}%")
    elif pred=='01':
        st.error(f"Good sit with the probability of {pred_prob[pred_idx]*100:.02f}%")
    elif pred=='02':
        st.warning(f"Unknow with the probability of {pred_prob[pred_idx]*100:.02f}%")

def main():
    st.title('Sitting Poseture Estimate')
    while True:
        img=cap.read()
        st.video(img[1], channels="BGR")
        time.sleep(10)
        predict(learn_inf, img[1])
        
if __name__ == '__main__':
    main()