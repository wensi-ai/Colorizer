import sys
sys.path.append(".")

import os
import streamlit as st
from importlib import import_module
from utils import v2i
from utils.util import apply_metric_to_video
from utils.metrics import *

@st.cache
def colorize(model, i):
    sys.argv = [sys.argv[0]]
    opt = TestOptions().parse()
    model.test("test/frames", f"test/output_{i}.mp4", opt)
    video = open(f"test/output_{i}.mp4", 'rb').read()
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
    ('DEVC', 'InstaColor')
)

# Add a slider to the sidebar:
input_method = st.sidebar.selectbox(
    'Select video input:',
    ("sample video", "random from dataset", "upload local video")
)

if input_method == "upload local video":
    video_org = st.sidebar.file_uploader("Choose a file")
    video_org = video_org.content()
elif input_method == "sample video":
    video_org = open('test/input.mp4', 'rb').read()
else:
    video_org = None

st.subheader("Original video:")
st.video(video_org, format="video/mp4", start_time=0)
st.subheader("Colorized video:")
st.subheader("Metrics:")
chart = st.line_chart([])

if st.sidebar.button('Colorize'):
    with open("test/temp.mp4", "wb") as f:
        f.write(video_org)
    video_in = v2i.convert_video_to_frames("test/temp.mp4", "test/frames")
    for i, model_name in enumerate(model_names):
        model_module = import_module(f"models.{model_name}")
        Model = getattr(model_module, model_name)
        opt_module = import_module(f"models.{model_name}.options.test_options")
        TestOptions = getattr(opt_module, "TestOptions")
        model = Model(pretrained=True)
        video_out = colorize(model, i)
        st.video(video_out, format="video/mp4", start_time=0)

        results = metric(f"test/output_{i}.mp4")
        for k in results:
            st.write(results[k])
    
    os.remove("test/temp.mp4")