import os
import torch
from utils.util import download_zipfile

class DEVC():
    def __init__(self, pretrained=False):
        if pretrained is True:
            download_zipfile("https://facevc.blob.core.windows.net/zhanbo/old_photo/colorization_checkpoint.zip", "DEVC_checkpoints.zip")
    def test(self): 
        pass
    def train(self):
        pass