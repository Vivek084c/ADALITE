import os
import pandas
# from src.logger import get_logger
# from src.custom_exception import CustomException
from utils.logger import get_logger
import yaml
import pandas as pd
import zipfile
import os
import gdown

logger = get_logger(__name__)

def read_yaml(filepath):
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File is not in the given path")
        with open(filepath, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Successfully read the yaml file")
            return config
    except Exception as e:
        logger.error("Error reading the YAML file")
        # raise CustomException("Failes to read the YAML file", e)
        


def download_drive_file_from_url(url, dest_folder):
    """
    Downloads a Google Drive file from a shareable URL and saves it in the specified folder.

    Args:
        url (str): Shareable Google Drive file URL (e.g., https://drive.google.com/file/d/FILE_ID/view?usp=sharing)
        dest_folder (str): Destination directory to save the file

    Returns:
        str: Path to the downloaded file
    """
    logger.info(f"Start -> downloading the file with url : {url} to destination : {dest_folder}")
    os.makedirs(dest_folder, exist_ok=True)

    # Use gdown to download
    output_path = gdown.download(url, quiet=False, fuzzy=True)

    if output_path is None:
        logger.error("Download failed. Check if the file is public or accessible.")
        raise Exception("Download failed. Check if the file is public or accessible.")

    # Move the downloaded file to the target folder
    file_name = os.path.basename(output_path)
    final_path = os.path.join(dest_folder, file_name)
    os.replace(output_path, final_path)

    logger.info(f"Done -> downloading the file with url : {url} to destination : {dest_folder}")
    return final_path



def extract_zip_file(zip_path, output_dir):
    """
    Extracts the contents of a .zip file to the specified output directory.

    Args:
        zip_path (str): Path to the .zip file
        output_dir (str): Directory where contents should be extracted

    Returns:
        None
    """
    logger.info(f"Start --> Extracting the files from the path {zip_path} to destination : {output_dir}")
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open and extract the .zip file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            logger.info(f"Extracting {zip_path} to {output_dir} ...")
            zip_ref.extractall(path=output_dir)
            logger.info(f"Done --> Extracting the files from the path {zip_path} to destination : {output_dir}")
    except zipfile.BadZipFile:
        logger.error("Error: Not a valid ZIP file.")
    except FileNotFoundError:
        logger.error(f"Error: File not found - {zip_path}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

def delete_zip_file(zip_path):
    """
    Deletes the specified .zip file if it exists.

    Args:
        zip_path (str): Path to the .zip file to be deleted

    Returns:
        bool: True if the file was deleted, False if not found
    """
    logger.info(f"Start --> Deleting the files from the path {zip_path}")
    if os.path.exists(zip_path) and zip_path.endswith('.rar'):
        try:
            os.remove(zip_path)
            logger.info(f"Deleted: {zip_path}")
            return True
        except Exception as e:
            logger.errors(f"Error deleting file: {e}")
            return False
    else:
        logger.info(f"File not found or not a zip: {zip_path}")
        return False
# def load_data(path):
#         """
#         function to load the data from a given path
#         """
#         try:
#             logger.info("Loading the data")
#             return pd.read_csv(path)
#         except Exception as e:
#             logger.error(f"Error loading the error {e}")
#             raise CustomException("Failed to load the data", e)