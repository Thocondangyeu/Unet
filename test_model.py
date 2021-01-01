from build_model import Unet
import cv2
import numpy as np
model=Unet("model.h5",(256,256,3))
image=cv2.imread("data_for_unet/test/49/images/49.jpg")

image=cv2.resize(image,(256,256))
image=image
image=image*1./255
image_expand=np.zeros((1,256,256,3))
image_expand[0]=image
result = model.predict(image_expand)
result=result>0.55
print(result.shape)
cv2.imshow("",(result[0]*255).astype(np.uint8))
cv2.waitKey()
for i in result[0]:
    print(i)