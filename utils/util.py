import os
import zipfile
import requests
import torch
from torchvision import io
import torchvideo.transforms as transforms

def download_zipfile(url: str, destination: str='.', unzip: bool=True):
    response = requests.get(url)
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    if unzip:
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            zip_ref.extractall(".")

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def apply_metric_to_video(video_path1, video_path2, metrics):
    batch_size = 50

    video1, _, _ = io.read_video(video_path1)
    video2, _, _ = io.read_video(video_path2)
    video1 = video1.cpu().detach().numpy()
    transform = transforms.Compose([
        transforms.NDArrayToPILVideo(),
        transforms.ResizeVideo((video2.shape[1], video2.shape[2])),
        transforms.CollectFrames(),
        transforms.PILVideoToTensor(rescale=False, ordering='TCHW')]
    )
    length = min(video1.shape[0], video2.shape[0])
    video1 = transform(video1)[:length].float()
    video2 = video2.permute((0, 3, 1, 2))[:length].float()
    results = []
    for metric in metrics:
        cur_metric_result = 0
        for i in range(0, video1.shape[0], batch_size):
            cur_metric_result += min(video1.shape[0] - i, batch_size) * metric(video1[i: min(video1.shape[0], i + batch_size)], video2[i: min(video1.shape[0], i + batch_size)])
        results.append(cur_metric_result / video1.shape[0])
    return results
