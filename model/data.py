from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import random
import numpy as np
import cv2
import albumentations as A


def expanded_image(image: np.ndarray, background_image: np.ndarray, x_expand_rate: float = 0.1, y_expand_rate: float = 0.1) -> Image:

    # Convert the images to numpy arrays
    # image_array = np.array(image)

    # Get the image width and height
    h, w, c = image.shape

    # Calculate the expansion range
    x_expand_range = int(0 * w), int(x_expand_rate * w)
    y_expand_range = int(0 * h), int(y_expand_rate * h)

    # Generate random expansion values for top, left, bottom, and right
    expand_top = random.randint(*y_expand_range)
    expand_bottom = random.randint(*y_expand_range)
    expand_left = random.randint(*x_expand_range)
    expand_right = random.randint(*x_expand_range)
    

    background_image = cv2.resize(background_image, (w+expand_left+expand_right, h+expand_top+expand_bottom))

    # Expand the image with the background image
    expanded_image = np.copy(background_image)
    expanded_image[expand_top:expand_top + h, expand_left:expand_left + w] = image

    # Convert the expanded image back to PIL image
    # expanded_image = Image.fromarray(expanded_image)

    return expanded_image



class CustomDataset(Dataset):
    def __init__(self, json_file: str, 
                       num_classes: int = 4, 
                       background_dir: str = None, 
                       transform = None):
        self.transform = transform

        # Load the JSON file
        with open(json_file, 'r') as f:
            data_json = json.load(f)
        
        self.classes = [str(i) for i in range(num_classes)] #data_json['classes']
        self.samples = data_json['samples']

        self.data_dir = os.path.dirname(json_file)
        self.num_classes = num_classes
        self.augment =A.Compose([
                                A.RandomBrightnessContrast(p=0.5),
                                A.GaussNoise(p=0.5),
                                A.Blur(p=0.5),
                                A.PixelDropout(dropout_prob = random.uniform(0.1, 0.5), p = 0.5),
                            ])
        if background_dir is None:
            self.bg_list = None
        else:
            self.bg_list = [cv2.cvtColor(cv2.imread(os.path.join(background_dir, x)), cv2.COLOR_BGR2RGB) for x in os.listdir(background_dir)]
    
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

        #augment
        image_array = np.array(image)
        if self.bg_list is not None:
            if random.uniform(0, 1) < 0.5:
                image_array = expanded_image(image_array, random.choice(self.bg_list), x_expand_rate = 0.05, y_expand_rate = 0.1)

    
        # Apply the augmentation transform to the image
        augmented = self.augment(image=image_array)

        # Retrieve the augmented image
        augmented_image = augmented["image"]

        # Convert the numpy array back to a PIL image
        augmented_image = Image.fromarray(augmented_image)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(augmented_image)
        
        return image, label

