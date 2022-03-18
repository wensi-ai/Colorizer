import os
import shutil
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transform_lib
import matplotlib.pyplot as plt

from utils.util import download_zipfile, mkdir
from utils.v2i import convert_frames_to_video

class OPT():
    pass

class DVP():
    def __init__(self):
        self.small = (320, 180)
        self.in_size = (0, 0)

    def test(self, black_white_path, colorized_path, output_path, opt=None):
        assert os.path.exists(black_white_path) and os.path.exists(colorized_path)
        
        self.downscale(black_white_path)
        self.downscale(colorized_path)

        os.system(f'python3 ./models/DVP/main_IRT.py --save_freq {opt.sf} --max_epoch {opt.me} --input {black_white_path} --processed {colorized_path} --model temp --with_IRT 1 --IRT_initialization 1 --output {opt.op}')

        frames_path = f"{opt.op}/temp_IRT1_initial1/{os.path.basename(black_white_path)}/00{opt.me}"

        self.upscale(frames_path)

        length = len(os.listdir(frames_path))
        frames = [f"out_main_{str(i).zfill(5)}.jpg" for i in range(length)]
        convert_frames_to_video(frames_path, output_path, frames)
        
    def downscale(self, path):
        frames = os.listdir(path)

        frame = Image.open(os.path.join(path, frames[0]))
        self.in_size = frame.size

        for each in frames:
            img = Image.open(os.path.join(path, each))
            img = img.resize(self.small, Image.ANTIALIAS)
            img.save(os.path.join(path, each))

    def upscale(self, path):
        frames = os.listdir(path)
        for each in frames:
            img = Image.open(os.path.join(path, each))
            img = img.resize(self.in_size, Image.ANTIALIAS)
            img.save(os.path.join(path, each))