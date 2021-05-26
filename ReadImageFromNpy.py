from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

filedir = r"D:\eternalaudrey\周宇\大学\大四上\实习\项目代码\python_projects\cyclegan\data\nature_image\raw\artifact"
fileindex = 276
filename = os.path.join(filedir,f"1.2.3.20210317.154436_0{fileindex}.npy")
img_array = np.load(filename)
plt.figure()
plt.imshow(img_array,cmap='gray')
plt.show()


