import os
import webdataset as wds
from torch.utils.data import DataLoader
import tarfile

from PIL import Image

def log_and_continue(err):
    if isinstance(err, tarfile.ReadError) and len(err.args) == 3:
        print(err.args[2])
        return True
    if isinstance(err, ValueError):
        return True
    raise err

def image_decoder(key, image_data):
    if key == ".jpg":
        return Image.open(io.BytesIO(image_data))
    return image_data

def text_decoder(key, text_data):
    if key == ".txt":
        return text_data.decode("utf-8")
    return text_data

class DatasetLoader:
    def __init__(self, path: str):
        self.path = path

        dataset = wds.WebDataset(path).decode(image_decoder, 
                                              text_decoder)

        all_samples = list(dataset)
        print(all_samples[0])