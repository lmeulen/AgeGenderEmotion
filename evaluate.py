"""
This application performs an age/gender/emotion estimation based upon a
video source (videofile or webcam)

Usage:
      python evaluate.py

For now, switching input source is done with global variables in the source code

The following work is used:
1. OpenCV2 haar cascade face recognition from https://github.com/opencv/opencv/
2. Gender and Age model                  from https://github.com/Tony607/Keras_age_gender
3. Emotion model                         from https://github.com/petercunha/Emotion
4. Wide Resnet implementation            from https://github.com/asmith26/wide_resnets_keras
"""
import numpy as np
from keras.models import load_model
import cv2
from wide_resnet import WideResNet
from utils.array import scale
from utils.image import crop_bounding_box, draw_bounding_box_with_label

# ResNet sizing
DEPTH = 16
WIDTH = 8

# Face image size
FACE_SIZE = 64

# Model location
# Face model from CV2 (Haar cascade)
FACE_MODEL_FILE = "models\\haarcascade_frontalface_alt.xml"
# Gender and Age model from https://github.com/Tony607/Keras_age_gender
# with Wide ResNet from https://github.com/asmith26/wide_resnets_keras
AG_MODEL_FILE = "models\\weights.18-4.06.hdf5"
# Emotion model from https://github.com/petercunha/Emotion
EM_MODEL_FILE = 'models\\emotion_model.hdf5'

# Source c onfiguration
USE_WEBCAM = False
WEBCAM_ID = 0
VIDEO_FILE = "demo.mp4"


def get_age_gender(face_image):
    """
    Determine the age and gender of the face in the picture
    :param face_image: image of the face
    :return: (age, gender) of the image
    """
    face_imgs = np.empty((1, FACE_SIZE, FACE_SIZE, 3))
    face_imgs[0, :, :, :] = face_image
    result = model.predict(face_imgs)
    est_gender = "F" if result[0][0][0] > 0.5 else "M"
    est_age = int(result[1][0].dot(np.arange(0, 101).reshape(101, 1)).flatten()[0])
    return est_age, est_gender


def get_emotion(face_image):
    """
    Determine the age and gender of the face in the picture
    :param face_image: image of the face
    :return: str:emotion of the image
    """
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray_face = scale(gray_face)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)

    # Get EMOTION
    emotion_prediction = emotion_classifier.predict(gray_face)
    emotion_probability = np.max(emotion_prediction)
    emotion_label_arg = np.argmax(emotion_prediction)
    return emotion_labels[emotion_label_arg]


# WideResNet model for Age and Gender
model = WideResNet(FACE_SIZE, depth=DEPTH, k=WIDTH)()
model.load_weights(AG_MODEL_FILE)

# VCC model for emotions
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad',
                  5: 'surprise', 6: 'neutral'}
emotion_classifier = load_model(EM_MODEL_FILE)
emotion_target_size = emotion_classifier.input_shape[1:3]

# Cascade model for face detection
face_cascade = cv2.CascadeClassifier(FACE_MODEL_FILE)

# Select video or webcam feed
if USE_WEBCAM:
    capture = cv2.VideoCapture(WEBCAM_ID)
else:
    capture = cv2.VideoCapture(VIDEO_FILE)

while capture.isOpened():
    success, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for face in faces:
        # Get face image, cropped to the size accepted by the WideResNet
        face_img, cropped = crop_bounding_box(frame, face, margin=.4, size=(FACE_SIZE, FACE_SIZE))
        (x, y, w, h) = cropped

        # Get AGE and GENDER and EMOTION
        (age, gender) = get_age_gender(face_img)
        emotion = get_emotion(face_img)

        # Add box and label to image
        label = "{}, {}, {}".format(age, gender, emotion)
        draw_bounding_box_with_label(frame, x, y, w, h, label)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
