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
batch = 8

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

im = np.random.rand(HEIGHT,WIDTH,3).astype(np.float16)
im = im.transpose((2, 0, 1))
im = np.random.rand(batch, 3, HEIGHT, WIDTH)


inf = Inference( serialized_plan_fp32, cuda_idx=0 )
inf.allocate_buffers(batch, trt.float32)


out = inf.do_inference( im, batch, HEIGHT, WIDTH)

model = keras.applications.resnet.ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
model.load_weights('model.h5')
print(out.sum())
import time
for i in range(500):
    imgs = np.random.rand(batch, 3, HEIGHT, WIDTH).astype(np.float32 ) #* np.random.randint(10)
    t = time.time()
    out = inf.do_inference( imgs, batch, HEIGHT, WIDTH )
    imgs = np.moveaxis( imgs, 1,-1).astype(np.float32)
    out2 = model.predict(imgs)
    #out2 = out.ravel()

    out = out.reshape(-1,1000)
    print('Out:',out2.argmax(axis=-1), out.argmax(axis=-1), out2.sum(), out.sum())
    
    t = time.time() - t
    
    print(int(1/t * batch), t)
