from stages.data_ingestion import download_data
from stages.load_model import MODEL

from utils.logger import get_logger
from utils.common_functions import read_yaml
logger = get_logger(__name__)


logger.info("Start --> Loading the yaml file")
config = read_yaml("config/config.yaml")
logger.info("Done  --> Loading the yaml file")


# ################################# WORKING FILE #################################
# logger.info(f"Start --> Data Ingestion ")
# download_data(
#     image_url = config["data_ingestion"]["image_url"] ,
#     destination = config["data_ingestion"]["destination_path"] ,
#     image_file_name = config["data_ingestion"]["image_file_name"]
#     )
# logger.info(f"Done --> Data Ingestion ")

# ################################# WORKING FILE #################################
# logger.info(f"Start --> Loading the model ")
# obj = MODEL("config/config.yaml")
# model = obj.load_model()
# logger.info(f"Done --> Loading the model ")







