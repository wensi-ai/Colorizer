import os
import cv2
from typing import List
import skvideo.io

def convert_video_to_frames(video_path: str, frames_path: str):
    """Convert video input to a set of frames in jpg format"""
    if os.path.isdir(frames_path) is False:
            os.makedirs(frames_path)
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    success = True
    success, image = vidcap.read()
    while success:
        cv2.imwrite(f"{frames_path}/frame" + str(count).zfill(5) + ".jpg", image)     # save frame as JPEG file
        if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            break
        success, image = vidcap.read()
        count += 1


def convert_frames_to_video(frames_path: str, output_video_path: str, images: List[str]=None):
    """Convert video input to a set of frames in jpg format"""
    if images is None:
        images = sorted(os.listdir(frames_path))
    frame = cv2.imread(os.path.join(frames_path, images[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter("temp.mp4", fourcc, 30.0, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(frames_path, image)))

    cv2.destroyAllWindows()
    video.release()

    os.system(f"ffmpeg -y -i temp.mp4 -vcodec libx264 {output_video_path}")
    os.remove("temp.mp4")