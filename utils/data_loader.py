import os
import cv2
import numpy as np
from utils.common_functions import save_tenosr_to_h5
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_data(
        IMAGE_PATH,
        interpreted,
        input_details,
        output_details,
        SOFTLABEL_DIR

):
    input_shape = input_details[0]['shape']
    inputHeight = input_shape[1]
    inputWidth = input_shape[2]

    output_shape = output_details[0]['shape']
    outputHeight = output_shape[1]
    outputWidth = output_shape[2]
    
    logger.info(f"Start --> Data generation using Teacher Model "
                f"(Input: {input_shape}, H={inputHeight}, W={inputWidth}; "
                f"Output: {output_shape}, H={outputHeight}, W={outputWidth})")
    #made changes
    # Make sure the image folder exists
    os.makedirs(IMAGE_PATH, exist_ok=True)    
    # imagePath = "midas/img/image.jpg"
    list_of_img = sorted([f for f in os.listdir(IMAGE_PATH) if f.endswith('.png')])
    logger.info(f"total images found  {len(list_of_img)}")
    for i,filename in enumerate(list_of_img):
        filepath = os.path.join(IMAGE_PATH, filename)
        ################################################################################# H5 : filename
        h5_filename = filename.split('.')[0]
        if i%10==0:
            logger.info(f"{i}th image is being processed, filename is {filename.split('.')[0]} and the path is {filepath} and it exist {os.path.exists(filepath)}")
        # imagePath = "midas/img/0000000000.png"
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        img_original = img
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_height, image_width, image_channel = img.shape
        # image_height, image_width, image_channel : 375, 1242, 3
        # Input values should be from -1 to 1 with a size of 128 x 128 pixels for the fornt model
        # and 256 x 256 pixels for the back model
        input_image = cv2.resize(img, (inputWidth, inputHeight), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        # inputWidth, inputHeight : 256, 256
        # Scale input pixel values to -1 to 1
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        reshaped_img = input_image.reshape(1, inputHeight, inputWidth, 3)    
        img_input = ((input_image/255.0 - mean) / std).astype(np.float32)
        img_input = img_input[np.newaxis, :, :, :]

    
        
        ################################################################################# H5 : input_img (with all preporcessing done) input shape is : (1, 256, 256, 3)
        h5_imput_img = img_input
        interpreted.set_tensor(input_details[0]['index'], img_input)
        interpreted.invoke()
        output = interpreted.get_tensor(output_details[0]['index'])
        ################################################################################# H5 : input_img (raw, no preprocessing done) output shape is (1, 256, 256, 1)
        h5_output_img = output
        # output shape is (1, 256, 256, 1)
        output = output.reshape(outputHeight, outputWidth)
        # outputHeight, outputWidth : 256, 256



        # we need output and img_original shape
        depth_min = output.min()
        depth_max = output.max()
        normlised_Depth_map = (255 * (output - depth_min) / (depth_max - depth_min)).astype('float32')

        #resize to the same size
        mapp = cv2.resize(normlised_Depth_map, (img_original.shape[1], img_original.shape[0]), interpolation=cv2.INTER_CUBIC)
        # mapp shape is -> (1092, 728)

        save_tenosr_to_h5(h5_imput_img, h5_output_img, h5_filename, SOFTLABEL_DIR)
    logger.info(f"Done --> Data generation using Teacher Model "
                f"(Input: {input_shape}, H={inputHeight}, W={inputWidth}; "
                f"Output: {output_shape}, H={outputHeight}, W={outputWidth})")