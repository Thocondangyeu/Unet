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
from utils.DataLoader import DataLoader

#mean_iou=tf.keras.metrics.MeanIoU(2, name=None, dtype=None)
# Set some parameters
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

model=Unet(None,(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))

TRAIN_PATH = "data_for_unet/train"#enter path to training data
TEST_PATH = "data_for_unet/test" #enter path to testing data

VALID_PATH = "data_for_unet/validation"

train_dir = glob.glob(TRAIN_PATH + "/*")
test_dir = glob.glob(TEST_PATH + "/*")
valid_dir = glob.glob(VALID_PATH + "/*")

#print(len(train_dir))




(X_val , y_val) = DataLoader(batch_size = 64 , dir =VALID_PATH).load_all()
# Build U-Net model


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
#model.summary()

#earlystopper = EarlyStopping(patience=5, verbose=1)
#checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
#results = model.fit(X_train/255, Y_train, validation_split=0.3, batch_size=16, epochs=50)
#model.save("Model/light-model.h5")
for i in range(30):
  data = DataLoader(dir  = TRAIN_PATH)
  for j in range(data.step):
    (X_train,y_train) = data.load()
    results = model.fit(X_train/255, y_train, validation_split=0.3, batch_size=16, epochs=50  , validation_data = (X_val , y_val))
                    #callbacks=[earlystopper, checkpointer])
  model.save("Model/light-model.h5")