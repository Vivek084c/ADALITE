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

def download_teacher_model(target_folder ):
    import shutil
    import os

    # Download first
    import kagglehub
    path = kagglehub.model_download("intel/midas/tfLite/v2-1-small-lite")
    logger.info(f"Downloaded to: {path}")

    # Define your target folder
    target_folder = "models/midas/"

    # Create it if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Copy files to your target folder
    model_path = ""
    for filename in os.listdir(path):
        source_file = os.path.join(path, filename)
        target_file = os.path.join(target_folder, filename)
        shutil.copy2(source_file, target_file)
        model_path = target_file

    logger.info(f"Copied to: {target_folder} and the model is {model_path}")
    return model_path

if __name__ == "__main__":
    pass
    
