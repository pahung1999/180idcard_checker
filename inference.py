from tqdm import tqdm
import yaml
import os
from torchvision import transforms
from PIL import Image
import torch
from model.model import SimpleModel, load_model
import random

yaml_path = "./config/train_config.yml"
with open(yaml_path, 'r') as file:
    config_gen = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model(model_name = config_gen['model_name'],
                   num_classes =  len(config_gen['classes']),
                   w = config_gen['fixed_w'], 
                   h = config_gen['fixed_h'])

model.load_state_dict(torch.load(config_gen['inference_weight']))
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((config_gen['fixed_w'], config_gen['fixed_h'])),  # Resize the image to a fixed size
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

image_folder = config_gen['image_folder']
for filename in os.listdir(image_folder):
    # Load and preprocess the test image
    test_image = Image.open(os.path.join(image_folder, filename)).convert('RGB')
    test_image = transform(test_image)
    test_image = test_image.unsqueeze(0)  # Add a batch dimension

    # Make predictions on the test image
    with torch.no_grad():
        outputs = model(test_image.to(device))
        _, predicted = torch.max(outputs, 1)
        
    # Get the predicted label
    predicted_label = config_gen["classes"][predicted.item()]

    print(f"{filename}")
    print(f'Predicted label: {predicted_label}')


# test_num = 20
# correct = 0
# for i in range(test_num):
#     filename = random.choice(os.listdir(image_folder))
#     image_path = os.path.join(image_folder, filename)

#     image = Image.open(image_path).convert("RGB")
#     if "180" in filename:
#         image = image.rotate(180, expand = True)
    
#     label = random.randint(0, 3)
#     image = image.rotate(int(90*label), expand=True)
    
#     test_image = transform(image)
#     test_image = test_image.unsqueeze(0)  # Add a batch dimension

#     # Make predictions on the test image
#     with torch.no_grad():
#         outputs = model(test_image.to(device))
#         _, predicted = torch.max(outputs, 1)
        
#     # Get the predicted label
#     predicted_label = config_gen["classes"][predicted.item()]

#     # print(f"{filename}")
#     print(f"Groundtruth: {label*90}")
#     print(f'Predicted label: {int(predicted.item())*90}')
#     print("="*15)