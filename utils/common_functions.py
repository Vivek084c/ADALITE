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
import tensorflow as tf
import h5py
import numpy as np
from datetime import datetime
import json

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
    
def save_tenosr_to_h5(input_image, output_image, filename, output_path):
    if isinstance(input_image, tf.Tensor):
        input_image = input_image.numpy()
    
    if isinstance(output_image, tf.Tensor):
        output_image = output_image.numpy()
    
    with h5py.File(output_path, 'a') as hf:
        # Create a group for each entry (use filename as the group name)
        if filename in hf:
            print(f"Skipping {filename}: already exists in H5 file.")
            return

        grp = hf.create_group(filename)
        grp.create_dataset("input_image", data=input_image)
        grp.create_dataset("output_image", data=output_image)
        grp.create_dataset("filename", data=np.bytes_(filename))


def silog_loss(y_true, y_pred, mask=None, lam=0.85, eps=1e-6):
    """
    Scale-invariant logarithmic loss for depth estimation with optional mask.

    Args:
        y_true: Tensor, reference depth map [batch, H, W, 1]
        y_pred: Tensor, predicted depth map [batch, H, W, 1]
        mask: Tensor of booleans [batch, H, W, 1] or [batch, H, W], optional
        lam: float, scale-invariant balancing factor
        eps: small number to avoid log(0)

    Returns:
        Scalar tensor: SiLog loss value
    """
    # Compute log difference
    log_diff = tf.math.log(y_pred + eps) - tf.math.log(y_true + eps)

    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        log_diff = log_diff * mask

        # Compute number of valid pixels to normalize properly
        valid_pixels = tf.reduce_sum(mask)
        mse_log = tf.reduce_sum(tf.square(log_diff)) / (valid_pixels + eps)
        mean_log = tf.reduce_sum(log_diff) / (valid_pixels + eps)
    else:
        mse_log = tf.reduce_mean(tf.square(log_diff))
        mean_log = tf.reduce_mean(log_diff)

    mean_log_sq = tf.square(mean_log)
    silog = tf.sqrt(mse_log - lam * mean_log_sq) * 10.0

    return silog


def convert_and_save_tflite(model_dir, tflite_path):
    import os
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
    tflite_model = converter.convert()

    # Ensure the output directory exists
    os.makedirs(tflite_path, exist_ok=True)

    tflite_path_model = os.path.join(tflite_path, "model.tflite")
    with open(tflite_path_model, "wb") as f:
        f.write(tflite_model)
        logger.info(f"Saved the tflite model : model.tflite")
    return tflite_path_model

def save_trained_model(STUDENT_MODEL_NEW, best_val_loss, config_file_path = "config/config.yaml"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = read_yaml(config_file_path)
    model_dir = config['model_training']['save_trained_model_dir']
    model_name = f"model_{timestamp}.h5"
    model_path = os.path.join(model_dir, model_name)

    logger.info(f"Start --> saving the model to path : {model_path}")
    STUDENT_MODEL_NEW.export(model_path)
    logger.info(f"Done --> saving the model : {model_path}")

    # ################################# update deployed_model.json  #################################

    deploy_info = {
        "model_path": model_path,
        "loss": float(best_val_loss),
        "timestamp": timestamp
    }
    deploy_json_path = os.path.join(model_dir, "deployed_model.json")

    # Load existing deployments if file exists, else start a new list
    if os.path.exists(deploy_json_path):
        with open(deploy_json_path, "r") as f:
            try:
                deployments = json.load(f)
                if not isinstance(deployments, list):
                    deployments = [deployments]
            except Exception:
                deployments = []
    else:
        deployments = []

    # Append new deployment info
    deployments.append(deploy_info)

    # Save all deployments back to the file
    with open(deploy_json_path, "w") as f:
        json.dump(deployments, f, indent=2)
    logger.info(f"Updated deployed_model.json at {deploy_json_path}")
    return timestamp



def update_deployment_model_loss(TRAINED_MODEL_LOSS, timestamp, config_path="config/deployment_config.yaml"):
    """
    Update deployment YAML file with the best loss and timestamp.
    Also, load the best model using the timestamp and save its TFLite version.
    """
    import json

    # 1. Read existing YAML config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # 2. Show current value to user
    DEPLOYED_MODEL_LOSS = config.get("deploynment", {}).get("deploynment_model_loss", None)
    print(f"Current deploynment_model_loss: {DEPLOYED_MODEL_LOSS}")

    if TRAINED_MODEL_LOSS < DEPLOYED_MODEL_LOSS:
        # 3. Update the YAML structure
        if "deploynment" not in config:
            config["deploynment"] = {}
        config["deploynment"]["deploynment_model_loss"] = TRAINED_MODEL_LOSS
        config["deploynment"]["timestamp"] = timestamp

        # 4. Write the updated config back to file
        with open(config_path, 'w') as file:
            yaml.safe_dump(config, file, sort_keys=False)
        print(f"✅ Updated deploynment_model_loss to {TRAINED_MODEL_LOSS} in {config_path}")

        # 5. Load the best model path from deployed_model.json using timestamp
        #    (Assume deployed_model.json is in the deployed model directory)
        config_main = read_yaml("config/config.yaml")
        model_dir = config_main['model_training']['save_trained_model_dir']
        deployed_json_path = os.path.join(model_dir, "deployed_model.json")
        best_model_path = None
        if os.path.exists(deployed_json_path):
            with open(deployed_json_path, "r") as f:
                deployments = json.load(f)
                for entry in deployments:
                    if entry.get("timestamp") == timestamp:
                        best_model_path = entry.get("model_path")
                        break

        if best_model_path is None:
            print(f"Could not find model with timestamp {timestamp} in {deployed_json_path}")
            return

        # 6. Save the TFLite version of the best model
        tflite_dir = config_main['model_training']['save_trained_model_tflite_dir']
        model_paht = convert_and_save_tflite(best_model_path, tflite_dir)
        print(f"Saved TFLite model at: {model_paht}")
    else:
        print("No update: the current model is the best model")
    
# update_deployment_model_loss("config/config.yaml")
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