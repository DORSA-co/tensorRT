import engine as eng
from inference  import Inference
from tensorflow import keras
import tensorrt as trt 
import numpy as np
import os


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


inf = Inference( serialized_plan_fp32, cuda_idx=0 )
inf.allocate_buffers(batch, trt.float32)


out = inf.do_inference( im, batch, HEIGHT, WIDTH)



print(out.sum())
import time
for i in range(50):
    imgs = np.random.rand(batch, 3, HEIGHT, WIDTH)
    t = time.time()
    out = inf.do_inference( im, batch, HEIGHT, WIDTH)
    t = time.time() - t
    print(int(1/t * batch), t)
