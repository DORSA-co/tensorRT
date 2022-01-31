import engine as eng
import inference as inf
from tensorflow import keras
import tensorrt as trt 
import numpy as np
import os
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

input_file_path = 'img.png'
onnx_file = "resnet50.onnx"
serialized_plan_fp32 = "resnet50.plan"
HEIGHT = 224
WIDTH = 224
batch = 64

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

im = np.random.rand(HEIGHT,WIDTH,3).astype(np.float32)
im = im.transpose((2, 0, 1))
im = np.random.rand(batch, 3, HEIGHT, WIDTH)


engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, batch, trt.float32)
print('run!')
out = inf.do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)
print(out)
pass



import time
for i in range(50):
    imgs = np.random.rand(batch, 3, HEIGHT, WIDTH)
    t = time.time()
    out = inf.do_inference(engine, imgs, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)
    t = time.time() - t
    print(int(1/t * batch), t)
