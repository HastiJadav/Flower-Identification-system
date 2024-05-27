# Flask utils
#tensorflow==2.15.0
from flask import Flask, request, render_template
import os
import cv2
import keras
import random
import numpy as np
import pandas as pd
import seaborn as sns
from os import listdir
from keras import layers
import tensorflow as tf
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import load_model
from keras.utils import plot_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
image_path = './static/images/inputimg.jpg'
default_image_size = tuple((128, 128))
image_size = 128

model =load_model('./Model/Cnn_Model.h5')

class_labels = pd.read_pickle('label_transform.pkl')
classes = class_labels.classes_

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}


def allowed_file(filename):
  return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home(): 
  return render_template('index.html')

@app.route('/backhome')
def backhome(): 
  return render_template('index.html')

@app.route('/imagerecognizerpage')
def imagerecognizerpage():
  return render_template('service.html', pred_class='none')

@app.route('/capimagerecognizerpage')
def capimagerecognizerpage():
  return render_template('servicewebcam.html', pred_class='none')

@app.route('/captureimg',methods=['POST'])
def captureimg():
  os.remove(image_path)
  cam = cv2.VideoCapture(0)
  #cv2.namedWindow("Capture Image")
  img_counter = 0
  
  while True:
      ret, frame = cam.read()
      if not ret:
          print("failed to grab frame")
          break
      cv2.imshow("test", frame)

      k = cv2.waitKey(1)
      if k % 256 == 8:
          # Press Backspace To Close=8
          print("Escape hit, closing...")
          break
      elif k % 256 == 32:
          # Press Spacebar To capture=32
          cv2.imwrite(image_path, frame)
          break

  cam.release()
  cv2.destroyAllWindows()

  return render_template('servicewebcam.html', pred_class='none', imagepath=image_path)


@app.route('/recognizedcap_flower',methods=['POST'])
def recognizedcap_flower():

  image = cv2.imread(image_path)
  image = img_to_array(image)
  image = cv2.resize(image, (image_size, image_size))
  image = np.array([image])
  prediction=model.predict(image)
  pred_= prediction[0]
  pred=[]
  for ele in pred_:
    pred.append(ele)
  maxi_ele = max(pred)
  idx = pred.index(maxi_ele)
  final_class=classes
  class_name= final_class[idx]
  class_text = "Recognized Flower is: " + class_name
  class_text = class_text.upper()


  return render_template('servicewebcam.html', imagepath=image_path,pred_class=class_text)


@app.route('/recognized_flower',methods=['POST'])
def recognized_flower():
  file = None
  file = request.files['file']
  #print(file)
  if file and allowed_file(file.filename):

    image = Image.open(file)
    image.save(os.path.join(image_path))
    image = cv2.imread(image_path)
    image = img_to_array(image)
    image = cv2.resize(image, (image_size, image_size))
    image = np.array([image])
    prediction=model.predict(image)
    pred_= prediction[0]
    pred=[]
    for ele in pred_:
      pred.append(ele)
    maxi_ele = max(pred)
    idx = pred.index(maxi_ele)
    final_class=classes
    class_name= final_class[idx]
    class_text = "Recognized Flower is: " + class_name
    class_text = class_text.upper()


  return render_template('service.html', imagepath=image_path,pred_class=class_text)


if __name__ == '__main__':
    app.run(debug=False)


