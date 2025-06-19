import tensorflow as tf
from tensorflow.keras import layers, models
from utils.logger import get_logger

logger = get_logger(__name__)

class DEPTH_MODEL(tf.keras.Model):
    def __init__(self, input_shape):
        logger.info(f"Start --> Creating DEPTH_MODEL with shape of {input_shape}")
        self.input_shape = input_shape
        logger.info(f"Done --> Creating DEPTH_MODEL with shape of {input_shape}")
        logger.info(f"Start --> Loading DEPTH_MODEL with shape of {input_shape}")
        self.model = self.build_model(self.input_shape)
        logger.info(f"Done --> Loading DEPTH_MODEL with shape of {input_shape}")

    def build_model(self, shape):
        # Input_block block 
        input_layers = layers.Input(shape = shape)

        output = layers.Conv2D(32, kernel_size=3, strides=2, padding="same")(input_layers)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(24, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)

        output = layers.Conv2D(144, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=True)(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(32, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "A")(output)
        A = output

        # # #################################################################

        output = layers.Conv2D(192, kernel_size=1, padding="valid")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(32, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "B")(output)
        output = layers.Add()([output, A])
        B = output

        # # #################################################################


        output = layers.Conv2D(192, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(32, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "C")(output)
        output = output = layers.Add()([output, B])
        C = output

        # # ############# INTERMDIATE FROM C  ---> 

        C = layers.Conv2D(64, kernel_size=3, padding="same")(C)
        C = layers.BatchNormalization()(C)
        C11 = C

        C = layers.ReLU()(C)

        C = layers.Conv2D(64, kernel_size=3, padding="same")(C)
        C = layers.BatchNormalization()(C)
        C = layers.ReLU()(C)

        C = layers.Conv2D(64, kernel_size=3, padding="same")(C)
        C = layers.BatchNormalization()(C)

        C1 = layers.Add()([C, C11])

        # # #################################################################

        output = layers.Conv2D(192, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=2, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(48, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "D")(output)
        D = output

        # #$ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ 

        output = layers.Conv2D(288, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(48, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "E")(output)
        output = layers.Add()([output, D])
        E = output

        # # #################################################################

        output = layers.Conv2D(288, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(48, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "F")(output)
        output = layers.Add()([output, E])
        F = output

        ##### INTERMDIATE FROM F  ---> after R block point

        F = layers.Conv2D(128, kernel_size=3, padding="same")(F)
        F = layers.BatchNormalization()(F)
        F11 = F

        F = layers.ReLU()(F)

        F = layers.Conv2D(128, kernel_size=3, padding="same")(F)
        F = layers.BatchNormalization()(F)
        F = layers.ReLU()(F)

        F = layers.Conv2D(128, kernel_size=3, padding="same")(F)
        F = layers.BatchNormalization()(F)

        F1 = layers.Add()([F, F11])


        # # #################################################################

        output = layers.Conv2D(288, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(96, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "G")(output)
        G = output

        # ##################################################################  4* #################################################################

        output = layers.Conv2D(576, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(96, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "H")(output)
        output = layers.Add()([output, G])
        H = output
        # #################################################################

        output = layers.Conv2D(576, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(96, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "I")(output)
        output = layers.Add()([output, H])
        I = output
        # #################################################################

        output = layers.Conv2D(576, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(96, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "J")(output)
        output = layers.Add()([output, I])
        J = output

        # #################################################################

        output = layers.Conv2D(576, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(96, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "K")(output)
        output = layers.Add()([output, J])
        K = output

        # ################################################################## #################################################################

        output = layers.Conv2D(576, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(136, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "L")(output)
        L = output

        # ################################################################## 4* #################################################################

        output = layers.Conv2D(816, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)


        output = layers.Conv2D(136, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "M1")(output)
        output = layers.Add()([output, L])
        M1 = output

        # ##################################################################

        output = layers.Conv2D(816, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)


        output = layers.Conv2D(136, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "M2")(output)
        output = layers.Add()([output, M1])
        M2 = output
        # ##################################################################

        output = layers.Conv2D(816, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)


        output = layers.Conv2D(136, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "M3")(output)
        output = layers.Add()([output, M2])
        M3 = output
        # ##################################################################

        output = layers.Conv2D(816, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(136, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "M4")(output)
        output = layers.Add()([output, M3])
        M3 = output

        ##### INTERMDIATE FROM M3  ---> 

        M3 = layers.Conv2D(256, kernel_size=3, padding="same")(M3)
        M3 = layers.BatchNormalization()(M3)

        M333 = M3

        M3 = layers.ReLU()(M3)

        M3 = layers.Conv2D(256, kernel_size=3, padding="same")(M3)
        M3 = layers.BatchNormalization()(M3)
        M3 = layers.ReLU()(M3)

        M3 = layers.Conv2D(256, kernel_size=3, padding="same")(M3)
        M3 = layers.BatchNormalization()(M3)

        M33 = layers.Add()([M3, M333])


        # ################################################################## * #################################################################

        output = layers.Conv2D(816, kernel_size=1, strides=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=2, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(232, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "N")(output)
        N = output

        # ################################################################## 5* #################################################################

        output = layers.Conv2D(1392, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(232, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "O1")(output)
        output = layers.Add()([output, N])
        O1 = output

        # ##################################################################

        output = layers.Conv2D(1392, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(232, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "O2")(output)
        output = layers.Add()([output, O1])
        O2 = output

        # ##################################################################

        output = layers.Conv2D(1392, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(232, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "O3")(output)
        output = layers.Add()([output, O2])
        O3 = output

        # ##################################################################

        output = layers.Conv2D(1392, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(232, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "O4")(output)
        output = layers.Add()([output, O3])
        O4 = output

        # ##################################################################

        output = layers.Conv2D(1392, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(232, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization(name = "O5")(output)
        output = layers.Add()([output, O4])
        O5 = output

        # ################################################################## * #################################################################

        output = layers.Conv2D(1392, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=True)(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(384, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)

        output = layers.Conv2D(512, kernel_size=3, padding="same")(output)
        output = layers.BatchNormalization()(output)
        P = output

        # ##################################################################

        output = layers.ReLU()(output)

        output = layers.Conv2D(512, kernel_size=3, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(512, kernel_size=3, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.Add()([output, P])

        # ##################################################################

        output = layers.Lambda(lambda img: tf.image.resize(img, size=(16, 16), method='bilinear'), output_shape=(16,16,512))(output)

        output = layers.Conv2D(256, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = output + M33
        Q = output

        # ##################################################################

        output = layers.ReLU()(output)

        output = layers.Conv2D(256, kernel_size=3, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(256, kernel_size=3, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.Add()([output,Q])
        R = output

        # ##################################################################

        output = layers.Lambda(lambda img: tf.image.resize(img, size=(32, 32), method='bilinear'),  output_shape=(32, 32, 256))(output)

        output = layers.Conv2D(128, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)

        output = layers.Add()([output, F1])
        output_1 = output

        # ##################################################################

        output = layers.ReLU()(output)

        output = layers.Conv2D(128, kernel_size=3, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(128, kernel_size=3, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.Add()([output,output_1])

        # ##################################################################

        output = layers.Lambda(lambda img: tf.image.resize(img, size=(64, 64), method='bilinear'),  output_shape=(64, 64, 128))(output)

        output = layers.Conv2D(64, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.Add()([output, C1])
        output_3 = output
        # ##################################################################

        output = layers.ReLU()(output)

        output = layers.Conv2D(64, kernel_size=3, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(64, kernel_size=3, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.Add()([output, output_3])

        # ##################################################################

        output = layers.Lambda(lambda img: tf.image.resize(img, size=(128, 128), method='bilinear'),  output_shape=(128, 128, 64))(output)

        output = layers.Conv2D(64, kernel_size=1, padding="same")(output)
        output = layers.BatchNormalization()(output)

        output = layers.Conv2D(32, kernel_size=3, padding="same")(output)
        output = layers.BatchNormalization()(output)

        output = layers.Lambda(lambda img: tf.image.resize(img, size=(256, 256), method='bilinear'), output_shape=(256, 256, 32))(output)

        output = layers.Conv2D(32, kernel_size=3, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        output = layers.Conv2D(1, kernel_size=3, padding="same")(output)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        STUDENT_MODEL = models.Model(input_layers, output)
        return STUDENT_MODEL

    def summary(self):
        return self.model.summary()
    
class TEACHER_MODEL:
    def __init__(self, model_path):
        logger.info("Start -> Loading the teacher model")
        self.model_path = model_path
        logger.info("Start -> building the teacher model")
        self.interpreted, self.input_details, self.output_details = self.build_model()
        logger.info("Done -> building the teacher model")
        logger.info("Done -> building the teacher model")

    def build_model(self):
        try:
            from tflite_runtime.interpreter import Interpreter
        except:
            from tensorflow.lite.python.interpreter import Interpreter
            
        print(f"Stating the data generation ")
        interpreted = Interpreter(model_path= self.model_path , num_threads=4)
        interpreted.allocate_tensors()

        #### get the details
        input_details = interpreted.get_input_details()
        input_shape = input_details[0]['shape']
        inputHeight = input_shape[1]
        inputWidth = input_shape[2]
        channels = input_shape[3]

        output_details = interpreted.get_output_details()
        output_shape = output_details[0]['shape']
        outputHeight = output_shape[1]
        outputWidth = output_shape[2]

        return interpreted, input_details, output_details
    
    
        
    
if __name__ == "__main__":
        # try:
        #     from tflite_runtime.interpreter import Interpreter
        # except:
        #     from tensorflow.lite.python.interpreter import Interpreter
            
        # print(f"Stating the data generation ")
        # interpreted = Interpreter(model_path= "models/midas/1.tflite" , num_threads=4)
        # interpreted.allocate_tensors()

        # #### get the details
        # input_details = interpreted.get_input_details()
        # input_shape = input_details[0]['shape']
        # inputHeight = input_shape[1]
        # inputWidth = input_shape[2]
        # channels = input_shape[3]

        # output_details = interpreted.get_output_details()
        # output_shape = output_details[0]['shape']
        # outputHeight = output_shape[1]
        # outputWidth = output_shape[2]

        # print("\n=== Inputs ===")
        # for i, d in enumerate(input_details):
        #     print(f"[{i}] name: {d['name']}, shape: {d['shape']}, dtype: {d['dtype']}")

        # print("\n=== Outputs ===")
        # for i, d in enumerate(output_details):
        #     print(f"[{i}] name: {d['name']}, shape: {d['shape']}, dtype: {d['dtype']}")
        pass