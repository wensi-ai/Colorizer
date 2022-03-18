import argparse
import torch.backends.cudnn as cudnn
class TestOptions():
    def __init__(self):
        self.sf = 10    # save frequency
        self.op = "test/results"
        self.me = 30    # max epoch

    def parse(self):
        return self