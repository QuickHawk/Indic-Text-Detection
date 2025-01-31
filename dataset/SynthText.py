import os
import scipy
import cv2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SynthText(Dataset):
    
    __URL = os.path.join("datasets", "SynthText", "data", "SynthText")
    
    def __init__(self, transform=None):
        self.transform = transform
        self.mat = scipy.io.loadmat(os.path.join(SynthText.__URL, 'gt.mat'))

    def __len__(self):
        return self.mat['imnames'].shape[-1]

    def __getitem__(self, idx):
        
        img_folder, img_name = self.mat['imnames'][0, idx][0].split('/')
        
        img_path = os.path.join(SynthText.__URL, img_folder, img_name)
        image = Image.open(img_path)
        width, height = image.size
        binary_map = np.zeros((height, width), dtype=np.uint8)
                    
        if len(self.mat['wordBB'][0, idx].shape) == 3:
            wordBB = np.transpose(self.mat['wordBB'][0, idx], (2, 1, 0))
        else:
            wordBB = np.expand_dims(self.mat['wordBB'][0, idx], axis=2)
            wordBB = np.transpose(wordBB, (2, 1, 0))
            
        cv2.fillPoly(binary_map, np.astype(wordBB, np.int32), [255, 255, 255])
            
        binary_map = Image.fromarray(np.astype(binary_map, np.uint8))
            
        if self.transform:
            image = self.transform(image)
            binary_map = self.transform(binary_map)
            
        return (image, binary_map)    
    
