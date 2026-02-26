import torch
import torch.nn as nn
import torchxrayvision as xrv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    model.to(DEVICE)
    model.eval()

    print("TorchXRayVision DenseNet-121 loaded successfully.")
    return model
