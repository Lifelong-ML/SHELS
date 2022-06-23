import glob
import os
import cv2 
import numpy as np
import torch 
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

class CustomDataset(Dataset):
    def __init__(self, data_path, transform):
        self.imgs_path = data_path
        file_list = glob.glob(self.imgs_path + "*")
        self.transform = transform 
        self.loader = default_loader
        self.data = []
        self.targets = []
        self.class_name_list = []
        i = 0
        self.class_map = {}
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            self.class_name_list.append(class_name)
            self.class_map[self.class_name_list[i]] = i
            for img_path in glob.glob(class_path+"/*.ppm"):
                self.data.append(img_path)
                self.targets.append(i)
            i+=1
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.classes = self.class_name_list 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        #path = os.path.join(sample)
        target = self.targets[idx]
        img = self.loader(path)
        if img is None:
            print(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target 



