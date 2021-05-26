import os
import os.path as path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import random
import shutil
import pydicom as pd

from tqdm import tqdm
from PIL import Image
from cyclegan.utils import read_dir, get_connected_components
from collections import defaultdict
from torchvision.utils import make_grid

ROOT_FOLDER = './data/nature_image/'
INPUT_FOLDER = './data/nature_image/raw/'
TRAIN_FOLDER = 'train/'
TEST_FOLDER = 'test/'
artifact_dir = 'artifact'
no_artifact_dir = 'no_artifact'
patient_dir = os.path.join(INPUT_FOLDER,no_artifact_dir)
'''
theFirstSlice = pd.dcmread(theFirst_dir) 
print(theFirstSlice.dir())
plt.figure(figsize=(10, 10))
plt.imshow(theFirstSlice.pixel_array, cmap=plt.cm.bone)
plt.show()
'''

if __name__ == '__main__':
    config_file = "config/dataset.yaml"
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['nature_image']
    image_size = config['image_size']
    if type(image_size) is not list: image_size = [image_size] * 2
    #for partient_dir in [os.path.join(ROOT_FOLDER,TRAIN_FOLDER,artifact_dir),
    #                    os.path.join(ROOT_FOLDER,TEST_FOLDER,artifact_dir),
    #                    os.path.join(ROOT_FOLDER,TRAIN_FOLDER,no_artifact_dir),
    #                    os.path.join(ROOT_FOLDER,TEST_FOLDER,no_artifact_dir)]:
    for patient_dir in [os.path.join(INPUT_FOLDER,artifact_dir),os.path.join(INPUT_FOLDER,no_artifact_dir)]:
        volume_files = read_dir(patient_dir,
                predicate=lambda x: x.endswith("dcm"), recursive=True)
        for i,volume_file in enumerate(volume_files):
            CTslice = pd.dcmread(volume_file)
            CTslice = CTslice.pixel_array.astype(np.float32)
            CTslice = Image.fromarray(CTslice).resize(image_size)
            CTslice_array = np.array(CTslice)
            CTslice = CTslice.convert("RGB")
            #save the npy file:
            if i < 400: my_dir = path.join(ROOT_FOLDER,TRAIN_FOLDER,path.basename(patient_dir))
            else: my_dir = path.join(ROOT_FOLDER,TEST_FOLDER,path.basename(patient_dir))
            image_array = path.join(my_dir,path.splitext(path.basename(volume_file))[0] + ".npy")
            image_file = path.join(my_dir,path.splitext(path.basename(volume_file))[0] + ".png")
            np.save(image_array,CTslice_array)
            CTslice.save(fp=image_file)