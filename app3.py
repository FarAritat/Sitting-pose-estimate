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
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
my_placeholder = st.empty()

MODEL_URL = "https://huggingface.co/spaces/farrr/Sitting-Poseture-Estimate/resolve/main/Sitting-Poseture-Estimate-model-1-final.pkl"
urllib.request.urlretrieve(MODEL_URL, "Sitting-Poseture-Estimate-model-1-final.pkl")
learn_inf = load_learner('Sitting-Poseture-Estimate-model-1-final.pkl')

def prepro():
    while run:
        _, frame = cap.read()
        FRAME_WINDOW.image(frame)
        predict(learn_inf, frame[1])
    else:
        st.write('Stopped')


def predict(learn, img):
    pred, pred_idx, pred_prob = learn.predict(img)
    if pred=='00':
        st.error(f"Bad sit with the probability of {pred_prob[pred_idx]*100:.02f}%")
    elif pred=='01':
        st.success(f"Good sit with the probability of {pred_prob[pred_idx]*100:.02f}%")
    elif pred=='02':
        st.warning(f"Unknow with the probability of {pred_prob[pred_idx]*100:.02f}%")

def main():
    #st.title('Sitting Poseture Estimate')
    prepro()
        
if __name__ == '__main__':
    main()