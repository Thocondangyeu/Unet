import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import cv2
from Model.build_model import Unet
import glob

#mean_iou=tf.keras.metrics.MeanIoU(2, name=None, dtype=None)
# Set some parameters
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

model=Unet(None,(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))

TRAIN_PATH = "/content/drive/MyDrive/Unet/data_for_unet/train"#enter path to training data
TEST_PATH = "/content/drive/MyDrive/Unet/data_for_unet/test" #enter path to testing data

VALID_PATH = "/content/drive/MyDrive/Unet/data_for_unet/validation"

train_dir = glob.glob(TRAIN_PATH + "/*")
test_dir = glob.glob(TEST_PATH + "/*")
valid_dir = glob.glob(VALID_PATH + "/*")

#print(len(train_dir))



class DataLoader :
  def __init__(self , batch_size = 32 ,dir = "/content/drive/MyDrive/Unet/data_for_unet/train"):
    self.batch_size = batch_size
    self.id =  0
    self.folder = glob.glob(dir + "/*")
    self.number = len(self.folder)
    self.step   = int(self.number / self.batch_size) + 1


  
  def load(self):
    N = len(train_dir)
    X_train = []
    y_train = []
    for i in range(self.id , self.id + self.batch_size):
      if i == N :
        self.id = 0 
        break
      data = self.folder[i]
      print("processing image{}".format(data))
      
      image = cv2.imread(data + "/image.jpg")
      if image is None :
        continue
      print("     {}".format(image.shape))
      image = cv2.resize(image ,(IMG_HEIGHT,IMG_WIDTH))

      mask  = cv2.imread(data + "/mask.png", 0)
      if mask is None : continue
      mask  = cv2.resize(mask ,(IMG_HEIGHT , IMG_WIDTH))

      print("     {}".format(mask.shape))

      X_train.append(image)
      y_train.append(mask)
    
    X_train = np.array(X_train).astype(np.float64) * 1./255
    y_train = (np.array(y_train) >0 ).astype(np.float64)
    return (X_train , y_train)
  def load_all(self):
    X_train = []
    y_train = []
    for data in self.folder:
      print("processing image{}".format(data))
      
      image = cv2.imread(data + "/image.jpg")
      if image is None :
        continue
      print("     {}".format(image.shape))
      image = cv2.resize(image ,(IMG_HEIGHT,IMG_WIDTH))

      mask  = cv2.imread(data + "/mask.png", 0)
      if mask is None : continue
      mask  = cv2.resize(mask ,(IMG_HEIGHT , IMG_WIDTH))

      print("     {}".format(mask.shape))

      X_train.append(image)
      y_train.append(mask)
      X_train = np.array(X_train).astype(np.float64) * 1./255
      y_train = (np.array(y_train) >0 ).astype(np.float64)
      return (X_train , y_train)

(X_val , y_val) = DataLoader(batch_size = 64 , dir =VALID_PATH).load_all()
# Build U-Net model


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
#model.summary()

#earlystopper = EarlyStopping(patience=5, verbose=1)
#checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
#results = model.fit(X_train/255, Y_train, validation_split=0.3, batch_size=16, epochs=50)
#model.save("Model/light-model.h5")
for i in range(30):
  data = DataLoader()
  for j in range(data.step):
    (X_train,y_train) = data.load()
    results = model.fit(X_train/255, y_train, validation_split=0.3, batch_size=16, epochs=50  , validation_data = (X_val , y_val))
                    #callbacks=[earlystopper, checkpointer])
  model.save("Model/light-model.h5")