from stages.data_ingestion import download_data, download_teacher_model
from stages.load_model import MODEL, TeacherModel, configure_gpu
from stages.generate_dataset import GENERATE_DATA, create_dataset
from stages.configure_optimizer_metrices import func_configure_optimizer_metrics_checkpoints
from utils.logger import get_logger
from utils.common_functions import read_yaml
logger = get_logger(__name__)


logger.info("Start --> Loading the yaml file")
config = read_yaml("config/config.yaml")
logger.info("Done  --> Loading the yaml file")


# ################################# Download the image data #################################
# logger.info(f"Start --> Data Ingestion ")
# download_data(
#     image_url = config["data_ingestion"]["image_url"] ,
#     destination = config["data_ingestion"]["destination_path"] ,
#     image_file_name = config["data_ingestion"]["image_file_name"]
#     )
# logger.info(f"Done --> Data Ingestion ")


# # ################################# Configure tenorflow to use GPU #################################
logger.info(f"Start --> Configure tenorflow to use GPU")
STUDENT_DEVICE = configure_gpu()
logger.info(f"Done --> Configure tenorflow to use GPU to {STUDENT_DEVICE}")


# ################################# Downloading the teacher model #################################
logger.info("Start -> Downloading the TEACHER Model")
model_path = download_teacher_model(config["teacher_model"]["target_folder"])
logger.info(f"Done -> Downloading the TEACHER Model to {model_path}")



# # ################################# Loading the teacher model #################################
logger.info(f"Start --> Loading the teacher model")
obj = TeacherModel("config/config.yaml")
student_model = obj.load_teacher_model(model_path)
interpreted, input_details, output_details = student_model.build_model()
logger.info(f"Done --> Loading the teacher model")



################################# Loading the studen modell #################################
logger.info(f"Start --> Loading the model ")
obj = MODEL("config/config.yaml")
model = obj.load_model()
logger.info(f"Done --> Loading the model ")


# ################################# Generating the dataset #################################
logger.info(f"Started the data generation")
GENERATE_DATA(
    IMAGE_PATH = config["data_ingestion"]["destination_path"],
    interpreted= interpreted,
    input_details= input_details,
    output_details= output_details,
    SOFTLABEL_DIR= config["data_generation"]["SOFTLABEL_DIR"]
)


# # ################################# Loading the dataset #################################
logger.info(f"Start --> Loading the dataest")

train_ds, val_ds, total_images = create_dataset(
    image_size = config["model_training"]["IMAGE_SIZE__"],
    batch_size = config["model_training"]["BATCH_SIZE"],
    softlabel_dir = config["model_training"]["SOFTLABEL"],
    split = config["model_training"]["TRAIN_VAL_SPLIT"]
)

logger.info(f"Done --> Loading the dataest total images found : {total_images}")



# # ################################# configure optimizer metric and checkpoint #################################
logger.info(f"Start --> Configuring the optimizer, metric and checkpoints")
lr_schedule, optimizer, train_metric, val_metric = func_configure_optimizer_metrics_checkpoints(
    TOTAL_IMGS= total_images,
    BATCH_SIZE= config["model_training"]["BATCH_SIZE"],
    STUDENT_MODEL= model,
    CHECKPOINT_DIR= config["model_training"]["CHECKPOINT_DIR"]
)
logger.info(f"Done --> Configuring the optimizer, metric and checkpoints")



# # ################################# configure optimizer metric and checkpoint #################################








