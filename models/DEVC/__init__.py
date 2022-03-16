import os
import shutil
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transform_lib

from utils.util import download_zipfile, mkdir
from utils.v2i import convert_frames_to_video
import models.DEVC.utils.lib.test_transforms as transforms
from models.DEVC.utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor
from models.DEVC.utils.util import batch_lab2rgb_transpose_mc, save_frames, tensor_lab2rgb, uncenter_l
from models.DEVC.models.ColorVidNet import ColorVidNet
from models.DEVC.models.FrameColor import frame_colorization
from models.DEVC.models.NonlocalNet import VGG19_pytorch, WarpNet

class DEVC():
    def __init__(self, pretrained=False):
        self.nonlocal_net = WarpNet(1)
        self.colornet = ColorVidNet(7)
        self.vggnet = VGG19_pytorch()

        if pretrained is True:
            download_zipfile("https://facevc.blob.core.windows.net/zhanbo/old_photo/colorization_checkpoint.zip", "DEVC_checkpoints.zip")
            self.vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))
            self.nonlocal_net.load_state_dict(torch.load("checkpoints/video_moredata_l1/nonlocal_net_iter_76000.pth"))
            self.colornet.load_state_dict(torch.load("checkpoints/video_moredata_l1/colornet_iter_76000.pth"))

    def test(self, input_path, output_path, opt):
        mkdir(opt.output_frame_path)
        # parameters for wls filter
        wls_filter_on = True
        lambda_value = 500
        sigma_color = 4

        # net
        self.nonlocal_net.eval()
        self.colornet.eval()
        self.vggnet.eval()
        self.nonlocal_net.cuda()
        self.colornet.cuda()
        self.vggnet.cuda()
        for param in self.vggnet.parameters():
            param.requires_grad = False

        # processing folders
        print("processing the folder:", input_path)
        _, _, filenames = os.walk(input_path).__next__()
        filenames.sort(key=lambda f: int("".join(filter(str.isdigit, f) or -1)))

        # NOTE: resize frames to 216*384
        transform = transforms.Compose(
            [CenterPad(opt.image_size), transform_lib.CenterCrop(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
        )

        # if frame propagation: use the first frame as reference
        # otherwise, use the specified reference image
        ref_name = os.path.join(input_path , filenames[0]) if opt.frame_propagate else opt.ref_path
        print("reference name:", ref_name)
        frame_ref = Image.open(ref_name)

        I_last_lab_predict = None

        IB_lab_large = transform(frame_ref).unsqueeze(0).cuda()
        IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")
        IB_l = IB_lab[:, 0:1, :, :]
        IB_ab = IB_lab[:, 1:3, :, :]
        with torch.no_grad():
            I_reference_lab = IB_lab
            I_reference_l = I_reference_lab[:, 0:1, :, :]
            I_reference_ab = I_reference_lab[:, 1:3, :, :]
            I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
            features_B = self.vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

        for index, frame_name in enumerate(tqdm(filenames)):
            frame1 = Image.open(os.path.join(input_path, frame_name))
            IA_lab_large = transform(frame1).unsqueeze(0).cuda()
            IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear")

            IA_l = IA_lab[:, 0:1, :, :]
            IA_ab = IA_lab[:, 1:3, :, :]
            
            if I_last_lab_predict is None:
                if opt.frame_propagate:
                    I_last_lab_predict = IB_lab
                else:
                    I_last_lab_predict = torch.zeros_like(IA_lab).cuda()

            # start the frame colorization
            with torch.no_grad():
                I_current_lab = IA_lab
                I_current_ab_predict, I_current_nonlocal_lab_predict, features_current_gray = frame_colorization(
                    I_current_lab,
                    I_reference_lab,
                    I_last_lab_predict,
                    features_B,
                    self.vggnet,
                    self.nonlocal_net,
                    self.colornet,
                    feature_noise=0,
                    temperature=1e-10,
                )
                I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)

            # upsampling
            curr_bs_l = IA_lab_large[:, 0:1, :, :]
            curr_predict = (
                torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 1.25
            )

            # filtering
            if wls_filter_on:
                guide_image = uncenter_l(curr_bs_l) * 255 / 100
                wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
                    guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), lambda_value, sigma_color
                )
                curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
                curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
                curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0)
                curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)
                curr_predict_filter = torch.cat((curr_predict_a, curr_predict_b), dim=1)
                IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
            else:
                IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])

            # save the frames
            save_frames(IA_predict_rgb, opt.output_frame_path, index)

        # output video
        convert_frames_to_video(opt.output_frame_path, output_path)

        shutil.rmtree("data")
        shutil.rmtree("checkpoints")

        print("Task Complete!")

    def train(self):
        pass