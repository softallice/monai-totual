from PIL import Image
import numpy as np
import os

# Batch convert image bit depth
path = r'./original/'
save_path = r'./change/'
for i in os.listdir(path):
    img = Image.open(path+i)
    img = Image.fromarray(np.uint8(img))
    t = img.convert('L')
    img = Image.fromarray(np.uint8(t))  # *255
    img.save(save_path+i)