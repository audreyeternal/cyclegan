import os.path as path
from .nature_image import NatureImage

def get_dataset(dataset_type, **dataset_opts):
    return {
        "nature_image": NatureImage
    }[dataset_type](**dataset_opts[dataset_type])
