#importing libraries
import tensorflow as tf
import dlib
import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

#Loading Of Model
model = load_model('drive/My Drive/model_deep.h5')

#Breaking the video into frames and making prediction
input_shape = (128, 128, 3)
x = []
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture('drive/My Drive/train_sample_videos/atkdltyyen.mp4')
frameRate = cap.get(5)
while cap.isOpened():
    frameId = cap.get(1)
    ret, frame = cap.read()
    if ret != True:
        break
    if frameId % ((int(frameRate)+1)*1) == 0:
        face_rects, scores, idx = detector.run(frame, 0)
        for i, d in enumerate(face_rects):
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            crop_img = frame[y1:y2, x1:x2]
            
            # print(crop_img)
            data = cv2.resize(crop_img, (128, 128))
            # x.append(data/255.0)
            # x=np.append(data/255.0)
            # print(x)
            # x = np.array(x)
            # print(x)
            # print(data)
            
            # print(img_to_array(cv2.resize(crop_img, (128, 128)))/255.0)
            data = np.array(img_to_array(cv2.resize(crop_img, (128, 128))).flatten() /255.0)
            data = data.reshape(-1, 128, 128, 3)
            # x = x.reshape(-1, 128, 128, 3)
            print(data)
            print(model.predict_classes(data))