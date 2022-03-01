import zipfile
import requests

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