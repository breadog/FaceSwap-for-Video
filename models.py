from face_68landmarks import face_68_landmarks
from face_enhancer import enhance_face
from face_recognizer import face_recognize
from face_swap import swap_face
from yolov8face import YOLOface_8n


def initialize_models():
    """初始化所有AI模型"""
    return {
        'detector': YOLOface_8n("weights/yoloface_8n.onnx"),
        'landmarker': face_68_landmarks("weights/2dfan4.onnx"),
        'encoder': face_recognize('weights/arcface_w600k_r50.onnx'),
        'swapper': swap_face('weights/inswapper_128.onnx'),
        'enhancer': enhance_face('weights/gfpgan_1.4.onnx')
    }


MODELS = initialize_models()