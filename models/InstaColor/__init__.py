import os
import shutil
import importlib
import numpy as np
import torch
import cv2
from typing import List
from tqdm import tqdm
from utils.util import download_pretrained_checkpoint

from models.InstaColor.models.base_model import BaseModel
from models.InstaColor.utils import util
from models.InstaColor.utils.datasets import *
from models.InstaColor.utils.download import download_zipfile

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

    def test(self, input_dir: str, file_names: List[str], opt):
        """
        Testing function 
        """
        #create bounding box save folder
        output_npz_dir = "{0}_bbox".format(input_dir)
        if os.path.isdir(output_npz_dir) is False:
            os.makedirs(output_npz_dir)
        
        # get bounding box for each image
        print("Getting bounding boxes...")
        for image_name in tqdm(file_names):
            img = cv2.imread(f"{input_dir}/{image_name}")
            lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab_image)
            l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)
            outputs = self.predictor(l_stack)
            pred_bbox = outputs["instances"].pred_boxes.to(torch.device('cpu')).tensor.numpy()
            pred_scores = outputs["instances"].scores.cpu().data.numpy()
            np.savez(f"{output_npz_dir}/{image_name.split('.')[0]}", bbox = pred_bbox, scores = pred_scores)

        #create colorized output save folder
        save_img_path = opt.results_img_dir
        if os.path.isdir(save_img_path) is False:
            os.makedirs(save_img_path)

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
                self.model.save_current_imgs(os.path.join(save_img_path, data_raw['file_id'][0] + '.png'))

        # remove bounding box dir
        shutil.rmtree(output_npz_dir)
        print("Task Complete")

    def _train_single_stage(self, opt):
        if opt.stage == 'full':
            dataset = Training_Full_Dataset(opt)
        elif opt.stage == 'instance':
            dataset = Training_Instance_Dataset(opt)
        elif opt.stage == 'fusion':
            dataset = Training_Fusion_Dataset(opt)
        else:
            print('Error! Wrong stage selection!')
            exit()
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)

        dataset_size = len(dataset)
        print('#training images = %d' % dataset_size)

        model = self.create_model(opt)
        model.setup(opt)

        opt.display_port = 8098
        visualizer = Visualizer(opt)
        total_steps = 0

        if opt.stage == 'full' or opt.stage == 'instance':
            for epoch in trange(opt.epoch_count, opt.niter + opt.niter_decay, desc='epoch', dynamic_ncols=True):
                epoch_iter = 0

                for data_raw in tqdm(dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
                    total_steps += opt.batch_size
                    epoch_iter += opt.batch_size

                    data_raw['rgb_img'] = [data_raw['rgb_img']]
                    data_raw['gray_img'] = [data_raw['gray_img']]

                    input_data = util.get_colorization_data(data_raw['gray_img'], opt, p=1.0, ab_thresh=0)
                    gt_data = util.get_colorization_data(data_raw['rgb_img'], opt, p=1.0, ab_thresh=10.0)
                    if gt_data is None:
                        continue
                    if(gt_data['B'].shape[0] < opt.batch_size):
                        continue
                    input_data['B'] = gt_data['B']
                    input_data['hint_B'] = gt_data['hint_B']
                    input_data['mask_B'] = gt_data['mask_B']

                    visualizer.reset()
                    model.set_input(input_data)
                    model.optimize_parameters()

                    if total_steps % opt.display_freq == 0:
                        save_result = total_steps % opt.update_html_freq == 0
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                    if total_steps % opt.print_freq == 0:
                        losses = model.get_current_losses()
                        if opt.display_id > 0:
                            visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

                if epoch % opt.save_epoch_freq == 0:
                    model.save_networks('latest')
                    model.save_networks(epoch)
                model.update_learning_rate()
        elif opt.stage == 'fusion':
            for epoch in trange(opt.epoch_count, opt.niter + opt.niter_decay, desc='epoch', dynamic_ncols=True):
                epoch_iter = 0

                for data_raw in tqdm(dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
                    total_steps += opt.batch_size
                    epoch_iter += opt.batch_size
                    box_info = data_raw['box_info'][0]
                    box_info_2x = data_raw['box_info_2x'][0]
                    box_info_4x = data_raw['box_info_4x'][0]
                    box_info_8x = data_raw['box_info_8x'][0]
                    cropped_input_data = util.get_colorization_data(data_raw['cropped_gray'], opt, p=1.0, ab_thresh=0)
                    cropped_gt_data = util.get_colorization_data(data_raw['cropped_rgb'], opt, p=1.0, ab_thresh=10.0)
                    full_input_data = util.get_colorization_data(data_raw['full_gray'], opt, p=1.0, ab_thresh=0)
                    full_gt_data = util.get_colorization_data(data_raw['full_rgb'], opt, p=1.0, ab_thresh=10.0)
                    if cropped_gt_data is None or full_gt_data is None:
                        continue
                    cropped_input_data['B'] = cropped_gt_data['B']
                    full_input_data['B'] = full_gt_data['B']
                    visualizer.reset()
                    model.set_input(cropped_input_data)
                    model.set_fusion_input(full_input_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
                    model.optimize_parameters()

                    if total_steps % opt.display_freq == 0:
                        save_result = total_steps % opt.update_html_freq == 0
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                    if total_steps % opt.print_freq == 0:
                        losses = model.get_current_losses()
                        if opt.display_id > 0:
                            visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
                if epoch % opt.save_epoch_freq == 0:
                    model.save_fusion_epoch(epoch)
                model.update_learning_rate()
        else:
            print('Error! Wrong stage selection!')
            exit()