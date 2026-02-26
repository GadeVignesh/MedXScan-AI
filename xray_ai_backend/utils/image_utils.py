import cv2
import numpy as np
import torch
import torchxrayvision as xrv

def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)
    img = img.astype(np.float32) / 255.0

    img = xrv.datasets.normalize(img, maxval=1)
    img = xrv.datasets.XRayResizer(224)(img)

    img = torch.from_numpy(img).unsqueeze(0)
    return img
