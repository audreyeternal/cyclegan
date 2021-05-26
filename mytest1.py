from matplotlib import cm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pydicom as pd
import scipy.io as sio
#image = np.load(r'D:\eternalaudrey\周宇\大学\大四上\实习\项目代码\python_projects\cyclegan\data\nature_image\train\no_artifact\1.2.3.20210317.153618_0072.npy')
#image = np.load(r'D:\eternalaudrey\周宇\大学\大四上\实习\项目代码\python_projects\cyclegan\data\nature_image\train\artifact\1.2.3.20210317.154436_0200.npy')
#image = np.load(r'D:\eternalaudrey\周宇\大学\大四上\实习\项目代码\python_projects\cyclegan\data\spineweb\train\artifact\patient0001_2805012\patient0001_2805012_024.npy')
#plt.hist(image.squeeze())
#plt.show()
#plt.imshow(image,cmap='gray')
#plt.show()
#image_seg= np.where(image<6000,image,0)
#plt.imshow(image_seg,cmap='gray')
#plt.show()
#image2 = Image.open(r"data/nature_image/train/artifact/1.2.3.20210317.154436_0056.jpg")
img = np.load('artifact_reduced.npy')
img_low = np.load('before_cyclegan.npy')
img_samedist = np.load('before_cyclegan.npy')
beforeCycleGAN = {"BeforeCycleGAN":img_low}
afterCycleGAN = {"AfterCycleGAN":img}
beforeCycleGAN_samedist = {"BeforeCycleGAN_samedist":img_samedist}
sio.savemat(r"D:\eternalaudrey\周宇\大学\大四上\实习\项目代码\matlab代码\NMAR\NMAR_CODE\after_cyclegan.mat",afterCycleGAN)
sio.savemat(r"D:\eternalaudrey\周宇\大学\大四上\实习\项目代码\matlab代码\NMAR\NMAR_CODE\before_cyclegan.mat",beforeCycleGAN)
sio.savemat(r"D:\eternalaudrey\周宇\大学\大四上\实习\项目代码\matlab代码\NMAR\NMAR_CODE\before_cyclegan_samedist.mat",beforeCycleGAN_samedist)