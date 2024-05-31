import os
import numpy as np
from PIL import Image
import onnx
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization import QuantType, QuantFormat, quantize_static, CalibrationDataReader
import onnxruntime as ort


def load_images_from_folder(folder, size=(640, 640)):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(size, Image.LANCZOS)
            img_data = np.array(img).astype(np.float32)
            img_data = np.transpose(img_data, (2, 0, 1))  # convert HWC to CHW
            img_data = img_data / 255.0  # normalize to [0, 1]
            img_data = np.expand_dims(img_data, axis=0)  # add batch dimension
            images.append(img_data)
    return images




class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_images, input_name):
        self.image_data = calibration_images
        self.data_index = 0
        self.input_name = input_name

    def get_next(self):
        if self.data_index < len(self.image_data):
            data = {self.input_name: self.image_data[self.data_index]}
            self.data_index += 1
            return data
        return None



# Load images
images = load_images_from_folder('/teamspace/studios/this_studio/larger_dataset/val/images/')
model_path = "/teamspace/studios/this_studio/yolov10/runs/detect/train2_10t_b16_e200_s640/weights/best_opset17.onnx"


model_prep_path = '/teamspace/studios/this_studio/quantized_yolo_models/pre_process/model_prep.onnx'

quant_pre_process(model_path, model_prep_path, skip_symbolic_shape=True)


model = onnx.load(model_prep_path)

# Create the ONNX runtime session to obtain input details
session = ort.InferenceSession(model_prep_path)
input_name = session.get_inputs()[0].name

# Create the calibration data reader
calibration_data_reader = MyCalibrationDataReader(images, input_name)



quantized_model_path = "/teamspace/studios/this_studio/quantized_yolo_models/yolov10t_b16_e200_s640_opset17_qint8_static.onnx"
quantize_static(
    model_prep_path, 
    quantized_model_path, 
    calibration_data_reader, 
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    )


