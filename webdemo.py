import sys
sys.path.append(".")

import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from importlib import import_module
from utils import v2i
from utils.util import apply_metric_to_video, mkdir
from utils.metrics import *
import shutil

mkdir("test")

@st.cache
def colorize(model, model_name, opt):
    sys.argv = [sys.argv[0]]
    model.test("test/frames", f"test/output_{model_name}.mp4", opt)
    video = open(f"test/output_{model_name}.mp4", 'rb').read()
    return video

@st.cache
def dvp(model, model_name, opt):
    sys.argv = [sys.argv[0]]
    model.test("test/frames", "test/results", f"test/output_{model_name}.mp4", opt)
    video = open(f"test/output_{model_name}.mp4", 'rb').read()
    return video

@st.cache
def metric(video_out):
    results = {}
    ret = apply_metric_to_video("test/temp.mp4", video_out, [PSNR, SSIM, LPIPS, cosine_similarity])
    results["PSNR"] = ret[0]
    results["SSIM"] = ret[1]
    results["LPIPS"] = ret[2]
    results["cosine_similarity"] = ret[3]
    return results

st.title("Video Colorization Web Demo")

# Add a selectbox to the sidebar:
model_names = st.sidebar.multiselect(
    'Select colorization model(s):',
    ('DEVC', 'InstaColor', 'Colorful')
)

# Add a slider to the sidebar:
input_method = st.sidebar.selectbox(
    'Select video input:',
    ("sample video", "random from dataset", "upload local video")
)
# get input video
if input_method == "upload local video":
    video_orig = st.sidebar.file_uploader("Choose a file")
    if video_orig:
        video_orig = video_orig.getvalue()
elif input_method == "sample video":
    video_orig = open('sample.mp4', 'rb').read()
else:
    video_orig = None

# display video
st.subheader("Original video:")
st.video(video_orig, format="video/mp4", start_time=0)
st.subheader("Colorized video:")

# setup metric
psnr, ssim, lpips, cs = [], [], [], []

if st.sidebar.button('Colorize'):
    with open("test/temp.mp4", "wb") as f:
        f.write(video_orig)
    video_in = v2i.convert_video_to_frames("test/temp.mp4", "./test/frames")
    for i, model_name in enumerate(model_names):
        # import model
        model_module = import_module(f"models.{model_name}")
        Model = getattr(model_module, model_name)
        # import and parse test options
        try:
            opt_module = import_module(f"models.{model_name}.options.test_options")
            TestOptions = getattr(opt_module, "TestOptions")
            opt = TestOptions().parse()
        except AttributeError: # no option needed
            opt = None
        # run test on model
        model = Model()
        if model_name == "DVP":
            video_out = dvp(model, model_name, opt)
        else:
            video_out = colorize(model, model_name, opt)
        st.write(model_name + ": ")
        st.video(video_out, format="video/mp4", start_time=0)
        print("video displayed")
        # get metrics
        results = metric(f"test/output_{model_name}.mp4")
        psnr.append(results["PSNR"])
        ssim.append(results["SSIM"])
        lpips.append(results["LPIPS"])
        cs.append(results["cosine_similarity"])
        print(f"{model_name} metric complete!")
    os.remove("test/temp.mp4")

shutil.rmtree("test/frames")
shutil.rmtree("test/results")

# display metrics
st.subheader("Metrics:")   
idx = model_names 
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 30))
# plt.subplots_adjust(wspace=1, hspace=1)
ax1.bar(idx, psnr, color=['grey' if (x < max(psnr)) else 'red' for x in psnr ], width=0.4)
ax2.bar(idx, ssim, color=['grey' if (x < max(ssim)) else 'red' for x in ssim ], width=0.4)
ax3.bar(idx, lpips, color=['grey' if (x > min(lpips)) else 'red' for x in lpips ], width=0.4)
ax4.bar(idx, cs, color=['grey' if (x < max(cs)) else 'red' for x in cs ], width=0.4)
ax1.set_title('PNSR (higher is better)')
ax2.set_title('SSIM (higher is better)')
ax3.set_title('LPIPS (higher is better)')
ax4.set_title('Cosine Similarity (higher is better)')
st.pyplot(fig)