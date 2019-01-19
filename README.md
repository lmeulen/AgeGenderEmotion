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

models directory must contain:
- emotion_model.hdf5
- haarcascade_frontalface_alt.xml
- weigts.18-4.06.hdf5

These can be obtained from the sources above