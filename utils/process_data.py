import os
import shutil
import glob 


data_dir = "HandSegData/hand_over_face"

image_dir = data_dir + "/images_resized"
mask_dir = data_dir + "/masks"

image_folder =  glob.glob(image_dir+"/*.jpg")
mask_folder = glob.glob(mask_dir+"/*.png")

num_image=len(image_folder)

os.mkdir("train")
os.mkdir("test")
print(num_image)
for i in range(int(num_image*0.8)):
    img_src = image_folder[i]
    mask_src = mask_folder[i]
    filename = (img_src.split("/")[-1]).split(".")[0].split("\\")[1]
    os.mkdir("train/"+filename)
    os.mkdir("train/"+filename+"/images/")
    os.mkdir("train/"+filename+"/masks/")
    img_dst="train/"+filename+"/images/"+filename+".jpg"
    mask_dst="train/"+filename+"/masks/"+filename+".png"
    shutil.copyfile(img_src,img_dst)
    shutil.copyfile(mask_src,mask_dst)

for i in range( int(num_image*0.8) , num_image ):
    img_src = image_folder[i]
    mask_src = mask_folder[i]
    filename = (img_src.split("/")[-1]).split(".")[0].split("\\")[1]
    os.mkdir("test/"+filename)
    os.mkdir("test/"+filename+"/images/")
    os.mkdir("test/"+filename+"/masks/")
    img_dst="test/"+filename+"/images/"+filename+".jpg"
    mask_dst="test/"+filename+"/masks/"+filename+".png"
    shutil.copyfile(img_src,img_dst)
    shutil.copyfile(mask_src,mask_dst)
