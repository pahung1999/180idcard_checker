from typing import Any
import torch.nn as nn

class SimpleModel(nn.Module):
    """
    Simple convolutional neural network model. Shouldn't use this model

    Args:
        class_num (int): The number of output classes. Default is 2.
        w (int): The width of the input image. Default is 1024.
        h (int): The height of the input image. Default is 1024.

    Attributes:
        features (nn.Sequential): The sequential module for the convolutional feature extraction layers.
        classifier (nn.Sequential): The sequential module for the classification layers.

    """
    def __init__(self, class_num = 2, w = 1024, h = 1024):
        super(SimpleModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Load the JSON file
        self.classifier = nn.Sequential(
            nn.Linear(32 * int(w/4) * int(h/4), 64),
            nn.ReLU(),
            nn.Linear(64, class_num)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_model(model_name: str,
               num_classes: int =2,
               **kwargs: Any):
    """
    Load a pre-defined model or custom model.

    Args:
        model_name (str): The name of the model to load.
        num_classes (int): The number of output classes. Default is 2.
        **kwargs (Any): Additional keyword arguments to pass to the model constructor.

    Returns:
        nn.Module: The loaded model.

    Raises:
        ValueError: If the specified model name is not supported.

    """
    if model_name == "SimpleModel":
        model = SimpleModel(class_num = num_classes, **kwargs)

    if model_name == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1, progress = True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    if model_name == "mobilenet_v3_large":
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1, progress = True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    
    return model
