'''
Models
ONNX
TorchScript
OpenVino
NCNN
'''

import numpy as np
import time
import torch
import onnxruntime as ort
from openvino import Core
from ultralytics import YOLO

# Try importing ncnn
try:
    import ncnn
    has_ncnn = True
except ImportError:
    has_ncnn = False
    print("⚠️ NCNN not installed. Skipping NCNN benchmark.")

# Create a dummy input
input_shape = (1, 3, 640, 640)
input_np = np.random.rand(*input_shape).astype(np.float32)
input_np_16 = np.random.rand(*input_shape).astype(np.float16)
input_tensor = torch.from_numpy(input_np)

# File paths
TORCHSCRIPT_PATH = r"F:\GP\Model Quantizatoin\PQT\YOLOv12 n\TorchScript\weights 12_CPU.torchscript"
ONNX_PATH = r"F:\GP\Model Quantizatoin\PQT\YOLOv12 n\ONNX\weights 12_CPU.onnx"
NCNN_PARAM_PATH = "model.param"
NCNN_BIN_PATH = "model.bin"
OPENVINO_XML_PATH = r"F:\GP\Model Quantizatoin\PQT\YOLOv12 n\OPEN_VENO\CPU\weights 12.xml"
YOLO_PATH= r"F:\weights 12.pt"
REPEAT = 1000

def benchmark_torchscript():
    print("\n🔥 TorchScript")
    model = torch.jit.load(TORCHSCRIPT_PATH)
    model.eval()
    with torch.no_grad():
        times = []
        for _ in range(REPEAT):
            start = time.time()
            _ = model(input_tensor)
            end = time.time()
            times.append((end - start) * 1000)
        print(f"Average Inference Time For TorchScript Model: {np.mean(times):.2f} ms")

def benchmark_onnx():
    print("\n🧠 ONNX Runtime")
    sess = ort.InferenceSession(ONNX_PATH)
    input_name = sess.get_inputs()[0].name
    times = []
    for _ in range(REPEAT):
        start = time.time()
        _ = sess.run(None, {input_name: input_np})
        end = time.time()
        times.append((end - start) * 1000)
    print(f"Average Inference Time For ONNX Model: {np.mean(times):.2f} ms")

def benchmark_openvino():
    print("\n🚀 OpenVINO")
    core = Core()
    model = core.read_model(OPENVINO_XML_PATH)
    compiled_model = core.compile_model(model, "CPU")
    input_layer = compiled_model.input(0)
    times = []
    for _ in range(REPEAT):
        start = time.time()
        _ = compiled_model([input_np])
        end = time.time()
        times.append((end - start) * 1000)
    print(f"Average Inference Time For OpenVino Model: {np.mean(times):.2f} ms")

def benchmark_ncnn():
    if not has_ncnn:
        return
    print("\n⚡ NCNN")
    net = ncnn.Net()
    net.load_param(NCNN_PARAM_PATH)
    net.load_model(NCNN_BIN_PATH)

    times = []
    for _ in range(REPEAT):
        ex = net.create_extractor()
        ex.set_light_mode(True)
        ex.set_num_threads(4)

        input_mat = ncnn.Mat.from_numpy(input_np[0])
        start = time.time()
        ex.input("input", input_mat)
        _, out = ex.extract("output")
        end = time.time()
        times.append((end - start) * 1000)

    print(f"Average Inference Time For NCNN Model: {np.mean(times):.2f} ms")
def benchmark_yolo():
    print("\n⚡ YOLO")
    input_np = (np.random.rand(640, 640, 3) * 255)
    model=YOLO(YOLO_PATH)
    times = []
    for _ in range(REPEAT):
        start = time.time()
        model.predict(input_np,verbose=False)
        end = time.time()
        times.append((end - start) * 1000)

    print(f"Average Inference Time For NCNN Model: {np.mean(times):.2f} ms")

if __name__ == "__main__":
    benchmark_torchscript()
    benchmark_onnx()
    benchmark_openvino()
    # benchmark_ncnn()
    benchmark_yolo()