import os
import requests
from zipfile import ZipFile
from urllib.request import urlopen



def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def download_google_drive_file(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)
    
def download_images():
    if not os.path.exists("images"):
        os.makedirs("images")
        print("Downloading DeepWeeds images to " + "./images/images.zip")
        download_google_drive_file("1bk_Fj6XaGvG5lqRlhQrwjSZBmUlmfGNL", "./images/images.zip")
        print("Finished downloading images.")
        print("Unzipping " + "./images/images.zip")
        with ZipFile("./images/images.zip", "r") as zip_ref:
            zip_ref.extractall("images")
        os.remove("./images/images.zip")
        print("Finished unzipping images.")
        
        
# Download images
download_images()
