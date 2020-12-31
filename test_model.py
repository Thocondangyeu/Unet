from build_model import Unet
import cv2

model=Unet("model-dsbowl2018-1.h5")
image=cv2.imread("/home/linhdt/Desktop/hieunk/Unet/data_for_unet/test/5/images/5.jpg")
image=cv2.resize(image,(256,256))

result = model.predict(image)
print(result)