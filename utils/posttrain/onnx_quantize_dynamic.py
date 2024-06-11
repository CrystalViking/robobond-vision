import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


#model_path = "/teamspace/studios/this_studio/yolov10/runs/detect/train11_xxs_e200_b16_s160/weights/best_opset17.onnx"
#model_path = "/teamspace/studios/this_studio/nanodet/minitower_nanodet_plux_m160_shufflev_x0_5_e200.onnx"
model_path = "/teamspace/studios/this_studio/nanodet/tdm_ndtpm_0_5x_320_e300_b16.onnx"
#model_path = "/teamspace/studios/this_studio/yolov10/runs/detect/train11_y10n_tde_yolo_orig_e200_b16_s320/weights/best.onnx"

onnx_model = onnx.load(model_path)


#quantized_model_path = "/teamspace/studios/this_studio/quantized_yolo_models/train11_y10n_tde_yolo_orig_e200_b16_s320_opset17_quint8_dynamic.onnx"
quantized_model_path = "/teamspace/studios/this_studio/quantized_nanodet_models/tdm_ndtpm_0_5x_320_e300_b16_quint8_dynamic.onnx"


'''
quantized_model = quantize_dynamic(
    model_path,
    quantized_model_path,
    weight_type=QuantType.QUInt8,
)
'''

quantized_model = quantize_dynamic(
    model_path,
    quantized_model_path,
    weight_type=QuantType.QUInt8,
)