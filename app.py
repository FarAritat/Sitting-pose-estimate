import streamlit as st
import numpy as np
import pickle
import cv2
import time
import urllib.request


pickle_in = open('Sitting-Poseture-Estimate-model-1-final.pkl', 'rb')
classifier = pickle.load(pickle_in)

st.text("Sitting-Poseture-Estimate")

# กำหนด URL ของภาพที่ต้องการโหลด
image_url = 'https://example.com/image.jpg'

# ดาวน์โหลดภาพจาก URL
urllib.request.urlretrieve(image_url, 'image.jpg')


    
video_capture = cv2.VideoCapture(0)
video_placeholder = st.empty()
last_capture_time = time.time()
interval = 10  # กำหนดช่วงเวลาในการเก็บภาพ

# Set the video frame size
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = video_capture.read()
    current_time = time.time()

    if current_time - last_capture_time >= interval:
        # เก็บภาพทุกๆ 10 วินาที
        last_capture_time = current_time
        # ทำตรงนี้สำหรับประมวลผลภาพหรือการทำ Machine Learning ต่อไป

    video_placeholder.image(frame, channels="BGR")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
prediction = classifier.predict(frame)
if prediction == 0:
    st.write("bad sit")
elif prediction == 1:
    st.write("good sit")
else:
    st.error("Cannot Analyze")
    st.info("Please sitting on a chair turn camera side view from your body")