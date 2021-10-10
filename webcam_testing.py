import cv2
from tensorflow import keras
import numpy as np


font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
color2 = (0, 0, 255)
thickness = 2

def main():
    model = keras.models.load_model("./models/model")
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #bgr to rgb
        img = cv2.resize(img, (64, 64)) #resize image
        img= np.array(img)
        img = img.astype(np.float32) / 255.  #normalize data
        img = img[np.newaxis, :]
        prediction = model.predict(img) #predict the output
        frame = cv2.flip(frame, 1)
        if prediction < 0.5: #if it is closer to 0, it is unsmiling
            cv2.putText(frame,"unsmiling", org, font,
                   fontScale, color2, thickness, cv2.LINE_AA)
        if prediction > 0.5: #if it is closer to 1, it is smiling
            cv2.putText(frame,"smiling",org, font,
                   fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow('my webcam', frame)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()