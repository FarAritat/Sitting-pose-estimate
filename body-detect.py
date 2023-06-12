import mediapipe as mp
import cv2

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

while True:
    cap = cv2.VideoCapture(1)
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # แปลงจาก BGR เป็น RGB
    mp_model = pose.process(imgRGB)
    if mp_model.pose_landmarks:
        mpDraw.draw_landmarks(img, mp_model.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(mp_model.pose_landmarks.landmark):
            h, w, c = img.shape
            # print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
