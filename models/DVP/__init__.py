
from .base_color import *
from .eccv16 import *
from .siggraph17 import *
from .util import *

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

class Colorful():
    def __init__(self, pretrained=True):
        self.model = siggraph17(pretrained=True).cuda().eval()
        self.opt = OPT()
        self.opt.output_frame_path = "./test/results"

    def test(self, input_path, output_path, opt=None):

        if not os.path.isdir(self.opt.output_frame_path):
            os.makedirs(self.opt.output_frame_path)
        
        frames = os.listdir(input_path)
        frames.sort()
        for frame in frames:
            colorized = self.colorize(os.path.join(input_path, frame))
            plt.imsave(os.path.join(self.opt.output_frame_path, frame), colorized)

        convert_frames_to_video(self.opt.output_frame_path, output_path)
        shutil.rmtree(self.opt.output_frame_path)
        

    def colorize(self, path):
        img = load_img(path)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
        tens_l_rs = tens_l_rs.cuda()
        out_img_siggraph17 = postprocess_tens(tens_l_orig, self.model(tens_l_rs).cpu())
        return out_img_siggraph17