from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, json_file, transform=None):
        self.transform = transform

        # Load the JSON file
        with open(json_file, 'r') as f:
            data_json = json.load(f)
        
        self.classes = data_json['classes']
        self.samples = data_json['samples']
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = os.path.join("./data", sample['image_path']) 
        label = int(sample['label'])

        # Load the image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label