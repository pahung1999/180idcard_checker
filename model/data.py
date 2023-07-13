from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import random
import numpy as np
import cv2
from model.augment import expanded_image, get_transform, rotate_img


class CustomDataset(Dataset):
    """
    Custom dataset class for loading and augmenting image data.

    Args:
        json_file (str): The path to the JSON file containing the dataset information.
        num_classes (int): The number of classes in the dataset. Default is 4.
        background_dir (str): The directory containing background images for augmentation. Default is None.
        transform (Callable): A callable function to apply transformations to the images. Default is None.

    Attributes:
        transform (Callable): A callable function to apply transformations to the images.
        classes (List[str]): The list of class labels.
        samples (List[Dict]): The list of sample data containing image paths and labels.
        data_dir (str): The directory containing the dataset files.
        num_classes (int): The number of classes in the dataset.
        augment (albumentations.Compose): The composition of image augmentations.
        bg_list (List[np.ndarray]): The list of background images for augmentation.

    """
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
        self.augment = get_transform(prob = 0.5)
        if background_dir is None:
            self.bg_list = None
        else:
            self.bg_list = [cv2.cvtColor(cv2.imread(os.path.join(background_dir, x)), cv2.COLOR_BGR2RGB) for x in os.listdir(background_dir)]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Get an item from the dataset at the specified index and augment with rotate, expand and some transforms from albumentations

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[Image.Image, int]: The augmented image and its corresponding label.

        """
        sample = self.samples[index]
        image_path = os.path.join(self.data_dir, sample['image_path']) 
        label = int(sample['label'])

        label = random.randint(0, self.num_classes-1)

        # Load the image
        image = Image.open(image_path)
        image = image.convert('RGBA')

        #Rotate augment
        if random.uniform(0, 1) < 0.5:
            image = rotate_img(image, random.randint(-10, 10)) 

        #Rotate label
        image = rotate_img(image, int(360*label/self.num_classes)) 

        #Expand with background
        if self.bg_list is not None:
            if random.uniform(0, 1) < 0.5:
                image = expanded_image(image, random.choice(self.bg_list), x_expand_rate = 0.1, y_expand_rate = 0.2)

        image = image.convert('RGB')
        image_array = np.array(image)

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

