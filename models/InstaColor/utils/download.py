#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
from importlib_metadata import requires
import requests
from os.path import join, isdir
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def download_coco_dataset(dataset_dir: str):
    print('download cocostuff training dataset')
    url = "http://images.cocodataset.org/zips/train2017.zip"
    response = requests.get(url, stream = True)
    if isdir(join(dataset_dir, "cocostuff")) is False:
        os.makedirs(join(dataset_dir, "cocostuff"))
    save_response_content(response, join(dataset_dir, "cocostuff", "train.zip"))
