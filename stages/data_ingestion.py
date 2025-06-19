import os

from utils.logger import get_logger
from utils.common_functions import download_drive_file_from_url, extract_zip_file, delete_zip_file



logger = get_logger(__name__)


def download_data(image_url, destination, image_file_name):
    # download the image datast
    logger.info(f"Start -> downloading the image dataset from the link : {image_url} destination : {destination} file name : {image_file_name}")
    image_file_url = image_url
    destination = destination
    download_drive_file_from_url(image_file_url, destination)
    logger.info(f"Done  -> downloading the image dataset from the link : {image_url} destination : {destination} file name : {image_file_name}")
    

    logger.info(f"Start -> Unzip the image dataset from the destination : {destination}")
    extract_zip_file(zip_path = os.path.join(destination, image_file_name), output_dir=destination)
    logger.info(f"Done -> Unzip the image dataset from the destination : {destination}")

    logger.info(f"Start -> Delete the image ZIP file name : {image_file_name}")
    delete_zip_file(os.path.join(destination, image_file_name))
    logger.info(f"Done -> Delete the image ZIP file name : {image_file_name}")

def download_teacher_model()
    
