from utils.logger import get_logger
from utils.common_functions import read_yaml
from models.DepthEstimationModel import DEPTH_MODEL

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, config_path):
        logger.info("Start --> Reading the YAML file")
        self.config = read_yaml(config_path)
        logger.info("Done --> Reading the YAML file")

    def load_model(self):
        logger.info("Start --> the model loading")
        model = DEPTH_MODEL(self.config["model_training"]["IMAGE_SIZE"])
        logger.info("Done --> the model loading")
        return model
    
    
    

if __name__ == "__main__":
    obj = ModelTraining("config/config.yaml")
    model = obj.load_model()
    model.summary()