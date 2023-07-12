from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import random

class CustomDataset(Dataset):
    def __init__(self, json_file, num_classes = 4 , transform=None):
        self.transform = transform

        # Load the JSON file
        with open(json_file, 'r') as f:
            data_json = json.load(f)
        
        self.classes = [str(i) for i in range(num_classes)] #data_json['classes']
        self.samples = data_json['samples']

        self.data_dir = os.path.dirname(json_file)
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = os.path.join(self.data_dir, sample['image_path']) 
        label = int(sample['label'])

        label = random.randint(0, self.num_classes-1)

        # Load the image
        image = Image.open(image_path).convert('RGB')
        image = image.rotate(int(360*label/self.num_classes), expand=True)
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label