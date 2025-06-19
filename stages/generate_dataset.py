from utils.data_loader import generate_data
from utils.common_functions import get_logger
import tensorflow as tf
import os
import numpy as np
import h5py

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


def create_dataset(
    image_size,
    batch_size,
    softlabel_dir,
    split,
    shuffle=True):
    logger.info("Start --> Data preprocessing and train val data generation")
    if not os.path.exists(softlabel_dir):
        logger.info(f"No softlabel file found: {softlabel_dir}")
        return None, None

    # Read keys only (no big arrays yet)
    with h5py.File(softlabel_dir, 'r') as hf:
        keys = list(hf.keys())
    total_images = len(keys)
    if shuffle:
        np.random.shuffle(keys)

    split_idx = int(len(keys) * split)
    train_keys = keys[:split_idx]
    val_keys = keys[split_idx:]

    def generator(keys_list):
        with h5py.File(softlabel_dir, 'r') as hf:
            for key in keys_list:
                input_image = hf[key]["input_image"][()]
                output_image = hf[key]["output_image"][()]
                
                # Remove extra leading dims
                input_image = np.squeeze(input_image)
                output_image = np.squeeze(output_image)
                
                input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
                output_image = tf.convert_to_tensor(output_image, dtype=tf.float32)
                
                # Ensure output_image has channel dim
                if output_image.ndim == 2:
                    output_image = tf.expand_dims(output_image, axis=-1)
                
                yield key.encode('utf-8'), input_image, output_image

    def make_tf_dataset(keys_list):
        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),  # name
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),  # input_image
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)   # output_image
        )

        ds = tf.data.Dataset.from_generator(
            lambda: generator(keys_list),
            output_signature=output_signature
        )
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_tf_dataset(train_keys)
    val_ds = make_tf_dataset(val_keys)

    logger.info("Done --> Data preprocessing and train val data generation")
    return train_ds, val_ds, total_images

    
    