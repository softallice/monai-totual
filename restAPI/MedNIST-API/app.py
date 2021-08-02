import io
import json

# import torchvision.models as models
import torch
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request

# monai module
from monai.networks.nets import DenseNet121
from monai.transforms import (
    AddChannel,
    Compose,
    ScaleIntensity,
    EnsureType
)

app = Flask(__name__)
mednist_class_index = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]

net = torch.jit.load("classifier.zip", map_location="cpu").eval()

def change_image(image_bytes):
    img = Image.fromarray(np.uint8(image_bytes))
    t = img.convert('L')
    img = Image.fromarray(np.uint8(t))
    img = img.resize((64, 64))
    return img


def transform_image(image_bytes):
    my_transforms = Compose([ AddChannel(), ScaleIntensity(), EnsureType()])
    
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.size == ('(64, 64)') :
        image = np.array(image)
    else:
        # 비트수준 8로 변환
        img = Image.fromarray(np.uint8(image))
        t = img.convert('L')
        img = Image.fromarray(np.uint8(t))
        img = img.resize((64, 64))
        image = np.array(img)

    print("input image", image.size)

    return my_transforms(image)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    print("================tensor===================== " )
    # print(tensor)
    with torch.no_grad():
        outputs = net(tensor[None].float())
        print("================outputs===================== " )
        print(outputs)

    _, output_classes = outputs.max(dim=1)
    print("================output_classes===================== " )
    # print(output_classes)

    # _, y_hat = outputs.max(1)
    # predicted_idx = str(y_hat.item())
    return mednist_class_index[output_classes[0]]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_name': class_name})


if __name__ == '__main__':
    app.run()