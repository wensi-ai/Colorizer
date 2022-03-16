import argparse
import torch.backends.cudnn as cudnn
class TestOptions():
    def __init__(self):
        self.initialized = False
        
    def initialize(self, parser):
        parser.add_argument(
            "--frame_propagate", default=True, type=bool, help="propagation mode, , please check the paper"
        )
        parser.add_argument("--image_size", type=int, default=[216 * 2, 384 * 2], help="the image size, eg. [216,384]")
        parser.add_argument("--cuda", action="store_false")
        parser.add_argument("--gpu_ids", type=str, default="0", help="separate by comma")
        parser.add_argument("--clip_path", type=str, default="../test/input.mp4", help="path of input clips")
        parser.add_argument("--ref_path", type=str, default="../test/frame00000.jpg", help="path of refernce images")
        parser.add_argument("--output_frame_path", type=str, default="../test/results", help="path of output colorized frames")
        
        self.initialized = True
        return parser

    def parse(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()

        opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(",")]
        cudnn.benchmark = True
        print("running on GPU", opt.gpu_ids)
        return opt