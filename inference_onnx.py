from tqdm import tqdm
import yaml
import os
from PIL import Image
import onnx
import onnxruntime
from PIL import Image
import numpy as np



yaml_path = "./config/train_config.yml"
with open(yaml_path, 'r') as file:
    config_gen = yaml.safe_load(file)

# Load the ONNX model
model = onnx.load("./weight/4labels/mobilenet_v3_small_lastest_w224_h224.onnx")

# Create an ONNX Runtime session
session_options = onnxruntime.SessionOptions()
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.enable_profiling = True  # Optional: Enable profiling for performance analysis

providers = ['CPUExecutionProvider']
session = onnxruntime.InferenceSession("./weight/4labels/mobilenet_v3_small_lastest_w224_h224.onnx", providers=providers)

image_folder = config_gen['image_folder']

for filename in os.listdir(image_folder):
    # Load and preprocess the test image
    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path).convert("RGB")
    image = image.resize((config_gen['fixed_w'], config_gen['fixed_h']))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    input_tensor = image_array[np.newaxis, ...]

    # Run the inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_tensor})

    # Get the predicted label
    output = output[0]
    predicted_label_index = np.argmax(output)
    print(f"{filename}: {int(predicted_label_index)*360/len(config_gen['classes'])}" )