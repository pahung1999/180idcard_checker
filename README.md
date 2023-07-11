# ID Card Rotation Check

This code is used to check if an ID card image is rotated 180 degrees or not.

## Available Models

The following models are available for rotation detection:

- SimpleModel with Conv2d
- mobilenet_v3_small
- mobilenet_v3_large

## Installation
Install the required dependencies. It is recommended to set up a virtual environment:
```shell
    pip install -r requirements.txt
```

## Usage
1. Prepare the training data:
    - Place ID card images in the ./images folder.
    - Create a JSON file (data.json) in the following format:

    ```json
    {
    "classes": ["0", "180"],
    "samples": [
        {
            "image_path": "./images/cccd_chip_back_0000.jpg",
            "label": "0"
        },
        {
            "image_path": "./images/cccd_chip_back_0001.jpg",
            "label": "1"
        },
        ...
    ]
    }
    ```

2. Configure the training settings: Modify the configuration file (./config/train_config.yml) to specify the model, hyperparameters, and training options.


3. Train the model: Run the training script:
```shell
    python train.py
```


4. Test the model: image folder and weight path in train_config.yml
```shell
    python inference.py
```
    
