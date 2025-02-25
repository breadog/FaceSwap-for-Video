# import onnxruntime
# print("可用提供者:", onnxruntime.get_available_providers())
# # 应输出 ['CUDAExecutionProvider', 'CPUExecutionProvider']

# import torch
#
# print(f"CUDA是否可用? {torch.cuda.is_available()}")
# print(f"当前CUDA 版本: {torch.version.cuda}")
#
# # Storing ID of current CUDA device
# cuda_id = torch.cuda.current_device()
# print(f"当前CUDA ID:{torch.cuda.current_device()}")
#
# print(f"CUDA设备名称:{torch.cuda.get_device_name(cuda_id)}")

# import onnxruntime
# print(onnxruntime.__version__)
# print(onnxruntime.get_device() ) # 如果得到的输出结果是GPU，所以按理说是找到了GPU的
# ort_session = onnxruntime.InferenceSession("weights/yoloface_8n.onnx",
# providers=['CUDAExecutionProvider'])
# print(ort_session.get_providers())

import onnxruntime

# 查看所有可用的 Execution Providers
print(onnxruntime.get_available_providers())

# 尝试强制使用 CUDA
try:
    ort_session = onnxruntime.InferenceSession(
        "weights/inswapper_128.onnx",
        providers=['CUDAExecutionProvider']
    )
    print("成功使用 CUDAExecutionProvider")
except Exception as e:
    print(f"无法使用 CUDAExecutionProvider: {e}")
