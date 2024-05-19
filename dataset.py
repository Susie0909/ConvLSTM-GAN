import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):  
    def __init__(self, root_dir, transforms_=None):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.transforms = transforms.Compose(transforms_)
        self.data = os.listdir(root_dir)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):  
        imgs_tensor_path = self.data[index] 
        imgs_tensor = torch.load(os.path.join(self.root_dir, imgs_tensor_path)) # tensor, [seq_len, h, w]
        if self.transforms:
            imgs_tensor = self.transforms(imgs_tensor)
        imgs_tensor = imgs_tensor.unsqueeze(1) # [seq_len, c, h, w]
        
        return imgs_tensor
