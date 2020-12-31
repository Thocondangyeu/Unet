from build_model import Unet
import cv2
import numpy as np
model=Unet("model-dsbowl2018-1.h5",(256,256,3))
image=cv2.imread("data_for_unet/test/49/images/49.jpg")

image=cv2.resize(image,(256,256))/255
image_expand=np.zeros((1,256,256,3))
image_expand[0]=image
result = model.predict(image_expand)
print(result.shape)
cv2.imshow("",(result[0]*255).astype(np.uint8))
cv2.waitKey()
print(result[0])