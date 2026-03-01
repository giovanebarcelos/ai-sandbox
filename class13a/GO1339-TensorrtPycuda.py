# GO1339-TensorrtPycuda
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Carregar engine TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)

with open('best.engine', 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Alocar buffers
input_shape = (1, 3, 640, 640)
output_shape = (1, 25200, 85)  # Depende do modelo

d_input = cuda.mem_alloc(np.prod(input_shape) * np.float32().itemsize)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.float32().itemsize)

# Inferência
cuda.memcpy_htod(d_input, img_batch)
context.execute_v2([int(d_input), int(d_output)])
output = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output, d_output)

# SPEEDUP TÍPICO: 2-5x vs PyTorch!
