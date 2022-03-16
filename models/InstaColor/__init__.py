import os
import shutil
import importlib
import numpy as np
import torch
import cv2
from typing import List
from tqdm import tqdm
from utils.util import download_zipfile, mkdir
from utils.v2i import convert_frames_to_video

from models.InstaColor.models.base_model import BaseModel
from models.InstaColor.utils import util
from models.InstaColor.utils.datasets import *
from utils.util import download_zipfile

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class InstaColor:
    def __init__(self, pretrained=False):
        # Bounding box predictor
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.predictor = DefaultPredictor(cfg)
        self.model = None

        # download pretrained model
        if pretrained is True:
            download_zipfile("https://docs.google.com/uc?export=download&id=1Xb-DKAA9ibCVLqm8teKd1MWk6imjwTBh&confirm=t", "InstaColor_checkpoints.zip")

    def find_model_using_name(self, model_name):
        # Given the option --model [modelname],
        # the file "models/modelname_model.py"
        # will be imported.
        model_filename = "models.InstaColor.models." + model_name + "_model"
        modellib = importlib.import_module(model_filename)

        # In the file, the class called ModelNameModel() will
        # be instantiated. It has to be a subclass of BaseModel,
        # and it is case-insensitive.
        model = None
        target_model_name = model_name.replace('_', '') + 'model'
        for name, cls in modellib.__dict__.items():
            if name.lower() == target_model_name.lower() \
            and issubclass(cls, BaseModel):
                model = cls

        if model is None:
            print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
            exit(0)

        return model


    def create_model(self, opt):
        model = self.find_model_using_name(opt.model)
        self.model = model()
        self.model.initialize(opt)
        print("model [%s] was created" % (self.model.name()))

    def test_images(self, input_dir: str, file_names: List[str], opt):
        """
        Testing function 
        """
        # get bounding box for each image
        print("Getting bounding boxes...")
        with torch.no_grad():
            for image_name in tqdm(file_names):
                img = cv2.imread(f"{input_dir}/{image_name}")
                lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab_image)
                l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)
                outputs = self.predictor(l_stack)
                pred_bbox = outputs["instances"].pred_boxes.to(torch.device('cpu')).tensor.numpy()
                pred_scores = outputs["instances"].scores.cpu().data.numpy()
                np.savez(f"{opt.output_npz_dir}/{image_name.split('.')[0]}", bbox = pred_bbox, scores = pred_scores)

        # setup dataset loader
        opt.batch_size = 1
        opt.test_img_dir = input_dir
        dataset = Fusion_Testing_Dataset(opt, file_names, -1)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size)

        # setup model to test
        self.create_model(opt)
        self.model.setup_to_test('coco_finetuned_mask_256_ffs')
        
        print("Colorizing...")
        # colorize image
        with torch.no_grad():
            for data_raw in tqdm(dataset_loader):
                data_raw['full_img'][0] = data_raw['full_img'][0].cuda()
                if data_raw['empty_box'][0] == 0:
                    data_raw['cropped_img'][0] = data_raw['cropped_img'][0].cuda()
                    box_info = data_raw['box_info'][0]
                    box_info_2x = data_raw['box_info_2x'][0]
                    box_info_4x = data_raw['box_info_4x'][0]
                    box_info_8x = data_raw['box_info_8x'][0]
                    cropped_data = util.get_colorization_data(data_raw['cropped_img'], opt, ab_thresh=0, p=opt.sample_p)
                    full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
                    self.model.set_input(cropped_data)
                    self.model.set_fusion_input(full_img_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
                    self.model.forward()
                else:
                    full_img_data = util.get_colorization_data(data_raw['full_img'], opt, ab_thresh=0, p=opt.sample_p)
                    self.model.set_forward_without_box(full_img_data)
                self.model.save_current_imgs(os.path.join(opt.results_img_dir, data_raw['file_id'][0] + '.png'))

    def test(self, input_dir: str, output_path: str, opt):
        # get file names
        input_files = os.listdir(input_dir)
        #create colorized output save folder       
        mkdir(opt.results_img_dir)
        #create bounding box save folder
        output_npz_dir = "{0}_bbox".format(input_dir)
        mkdir(output_npz_dir)
        opt.output_npz_dir = output_npz_dir

        for i in range(0, len(input_files), 5):
            self.test_images(input_dir, input_files[i : min(len(input_files), i + 5)], opt)
        
        output_frames = sorted(os.listdir(opt.results_img_dir))
        # resize back
        for output in output_frames:
            img = Image.open(os.path.join(opt.results_img_dir, output))
            img = img.resize((320, 180), Image.ANTIALIAS)
            img.save(os.path.join(opt.results_img_dir, output))
        
        convert_frames_to_video(opt.results_img_dir, output_path)
        # remove bounding box dir
        shutil.rmtree(output_npz_dir)
        shutil.rmtree(opt.results_img_dir)
        print("Task Complete")