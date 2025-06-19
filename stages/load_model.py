from utils.logger import get_logger
from utils.common_functions import read_yaml, save_tenosr_to_h5
from models.DepthEstimationModel import DEPTH_MODEL, TEACHER_MODEL
import tensorflow as tf
logger = get_logger(__name__)

class MODEL:
    def __init__(self, config_path):
        logger.info("Start --> Reading the YAML file")
        self.config = read_yaml(config_path)
        logger.info("Done --> Reading the YAML file")

    def load_model(self):
        logger.info("Start --> the model loading")
        model = DEPTH_MODEL(self.config["model_training"]["IMAGE_SIZE"])
        logger.info("Done --> the model loading")
        return model
    
class TeacherModel:
    def __init__(self, config_path):
        logger.info("Start --> Reading the YAML file")
        self.config = read_yaml(config_path)
        logger.info("Done --> Reading the YAML file")
    
    def load_teacher_model(self, model_path):
        logger.info(f"Start --> loading the teacher model")
        model = TEACHER_MODEL(model_path)
        logger.info("Done -> loading the teacher model")
        return model
    
def configure_gpu():
    try:
        tf_gpus = tf.config.list_physical_devices('GPU')
        if tf_gpus:
            tf.config.set_visible_devices(tf_gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(tf_gpus[0], True)
            STUDENT_DEVICE = '/GPU:0'  # TensorFlow device string
            print(f"✅ TensorFlow assigned to GPU 0: {STUDENT_DEVICE}")
        else:
            STUDENT_DEVICE = '/CPU:0'
            print("⚠️ No GPU found for TensorFlow. Using CPU.")
    except Exception as e:
        STUDENT_DEVICE = '/CPU:0'
        print(f"❌ TensorFlow device setup failed: {e}. Using CPU.")
    return STUDENT_DEVICE


if __name__ == "__main__":
    obj = MODEL("config/config.yaml")
    model = obj.load_model()
    model.summary()