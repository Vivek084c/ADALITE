from utils.logger import get_logger
import tensorflow as tf

logger = get_logger(__name__)

def train_model():
    # Setup TensorFlow to use GPU 0
    logger.info(f"Start --> Configuring the GPU for Training")
    try:
        tf_gpus = tf.config.list_physical_devices('GPU')
        if tf_gpus:
            tf.config.set_visible_devices(tf_gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(tf_gpus[0], True)
            STUDENT_DEVICE = '/GPU:0'  # TensorFlow device string
            logger.info(f"✅ TensorFlow assigned to GPU 0: {STUDENT_DEVICE}")
        else:
            STUDENT_DEVICE = '/CPU:0'
            logger.info("⚠️ No GPU found for TensorFlow. Using CPU.")
    except Exception as e:
        STUDENT_DEVICE = '/CPU:0'
        logger.info
        (f"❌ TensorFlow device setup failed: {e}. Using CPU.")
    logger.info(f"Done --> Configuring the GPU for Training")

    
