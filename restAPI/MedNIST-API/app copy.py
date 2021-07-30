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
    LoadImage,
    ScaleIntensity,
    EnsureType,
    ToTensor
)

app = Flask(__name__)
mednist_class_index = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]
# model = models.DenseNet121(pretrained=False)
device = torch.device("cpu")
model = DenseNet121(
    spatial_dims=2,
    in_channels=1,
    out_channels=len(mednist_class_index) 
).to(device)
model.load_state_dict(torch.load('best_metric_model.pth'))
model.eval()


def transform_image(image_bytes):
    # my_transforms = Compose([ AddChannel(), ScaleIntensity(), EnsureType()])
    my_transforms = Compose([
        LoadImage(image_only=True),
        AddChannel(),
        ScaleIntensity(),
        ToTensor()
    ])

    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    return my_transforms(image)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor = tensor.to(device)
    print("================tensor===================== " )
    print(tensor)
    with torch.no_grad():
        outputs = model(tensor).argmax(dim=0)
        print("================outputs===================== " )
        print(outputs)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
    return mednist_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()