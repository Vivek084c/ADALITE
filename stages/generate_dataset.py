from utils.data_loader import generate_data
from utils.common_functions import get_logger

logger = get_logger(__name__)

def GENERATE_DATA(
        IMAGE_PATH,
        interpreted,
        input_details,
        output_details,
        SOFTLABEL_DIR
):
    logger.info(f"Start --> Data generation from Teacher Models IMAGE_PATH : {IMAGE_PATH}, SOFTLABEL_DIR : {SOFTLABEL_DIR}")
    generate_data(
        IMAGE_PATH = IMAGE_PATH,
        interpreted = interpreted,
        input_details = input_details,
        output_details = output_details,
        SOFTLABEL_DIR = SOFTLABEL_DIR
           
    )
    logger.info(f"Done --> Data generation from Teacher Models IMAGE_PATH : {IMAGE_PATH}, SOFTLABEL_DIR : {SOFTLABEL_DIR}")
    
    