import os
import pandas
# from src.logger import get_logger
# from src.custom_exception import CustomException
from utils.logger import get_logger
import yaml
import pandas as pd

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