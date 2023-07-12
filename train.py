import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
from model.data import CustomDataset
from model.model import SimpleModel
from tqdm import tqdm
import yaml
import os
from PIL import Image
import albumentations as A



MODEL_LIST = ["SimpleModel", "mobilenet_v3_small", "mobilenet_v3_large"]

yaml_path = "./config/train_config.yml"
with open(yaml_path, 'r') as file:
    config_gen = yaml.safe_load(file)

fixed_w = config_gen['fixed_w']
fixed_h = config_gen['fixed_h']
num_classes = len(config_gen['classes'])

if not os.path.exists(config_gen['weight_dir']):
   os.makedirs(config_gen['weight_dir'])

transform = transforms.Compose([
    transforms.Resize((fixed_h, fixed_w)),  # Resize the image to a fixed size
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

#Load data
print("Creating dataset...")
train_dataset = CustomDataset(config_gen['train_path'], 
                              num_classes = num_classes, 
                              background_dir = config_gen['background_dir'], 
                              transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config_gen['batch_size'], shuffle=True)

val_dataset = CustomDataset(config_gen['val_path'], 
                            num_classes = num_classes, 
                            background_dir = config_gen['background_dir'], 
                            transform=transform)
val_loader = DataLoader(val_dataset, batch_size=config_gen['batch_size'], shuffle=False)


#Load model
if config_gen['model_name'] not in MODEL_LIST:
    raise ValueError(f"Can't find {config_gen['model_name']} in MODEL_LIST = {MODEL_LIST}")
else:
    if config_gen['model_name'] == "SimpleModel":
        model = SimpleModel(class_num = num_classes, w = fixed_w, h = fixed_h)

    if config_gen['model_name'] == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1, progress = True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    if config_gen['model_name'] == "mobilenet_v3_large":
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1, progress = True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

if config_gen["pre_trained"] is not None:
    pretrained_path = config_gen["pre_trained"]
    model.load_state_dict(torch.load(pretrained_path))
    print(f"Loaded weight: {pretrained_path}")
    print("-"*15)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

best_acc = 0
num_epochs = config_gen['epochs']
# Create an example input tensor
example_input = torch.randn(1, 3, fixed_h, fixed_w).to(device)
print("Training process....")
for epoch in range(num_epochs):
    print("Epoch: ",epoch+1)
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}')

    model.eval()
    val_loss = 0.0
    correct = 0

    print("Validating...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_dataset)
    accuracy = correct / len(val_dataset)
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

    weight_path = os.path.join(config_gen['weight_dir'], f"{config_gen['model_name']}_lastest_w{fixed_w}_h{fixed_h}.pth")
    torch.save(model.state_dict(), weight_path)

    onnx_path = os.path.join(config_gen['weight_dir'], f"{config_gen['model_name']}_lastest_w{fixed_w}_h{fixed_h}.onnx")
    # Export the model to ONNX format
    torch.onnx.export(model, example_input, onnx_path)

    print(f"Saved last weight to: {weight_path}")
    print(f"Saved last weight to: {onnx_path}")

    print(f"Best Accuracy: {best_acc}")
    if accuracy > best_acc:
        best_acc = accuracy
        weight_path = os.path.join(config_gen['weight_dir'], f"{config_gen['model_name']}_best_acc_w{fixed_w}_h{fixed_h}.pth")
        torch.save(model.state_dict(), weight_path)

        onnx_path = os.path.join(config_gen['weight_dir'], f"{config_gen['model_name']}_best_acc_w{fixed_w}_h{fixed_h}.onnx")
        # Export the model to ONNX format
        torch.onnx.export(model, example_input, onnx_path)

        print(f"Saved best weight to: {weight_path}")
        print(f"Saved best weight to: {onnx_path}")
