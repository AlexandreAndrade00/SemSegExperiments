import numpy as np
import torch
from PIL import Image
from os.path import splitext


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename)).convert('RGB')
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy()).convert('RGB')
    else:
        return Image.open(filename).convert('RGB')
