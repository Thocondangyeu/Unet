class DataLoader :
  def __init__(self , batch_size = 32 ,dir = "../data_for_unet/train"):
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
