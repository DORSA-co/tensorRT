import engine as eng
import argparse
from onnx import ModelProto
import tensorrt as trt 
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"






import pycuda.driver as cuda
import pycuda.autoinit 

# class A:

#    def __init__(self, cuda_idx=0):

#       #cuda_idx = 0
#       self.cuda_idx = cuda_idx
#       print('Selected Device is {}'.format(cuda.Device( self.cuda_idx ).name()))
#       self.cfx = cuda.Device( self.cuda_idx ).make_context()
#       self.stream = cuda.Stream()


#       TRT_LOGGER = trt.Logger(trt.Logger.INFO)
#       trt.init_libnvinfer_plugins(TRT_LOGGER, '' )
#       trt_runtime = trt.Runtime(TRT_LOGGER)

      
# a = A(1)    


def ebuilder(onnx_path, engine_path, engine_name, batch_size, cuda_idx=0):
    print('Build Engine on {}'.format(cuda.Device( cuda_idx ).name()))
    cfx = cuda.Device( cuda_idx ).make_context()
    stream = cuda.Stream()




    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())

    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    shape = [batch_size , d0, d1 ,d2]
    engine = eng.build_engine(onnx_path, shape= shape, precision='fp16')

    engine_name = '{}_{}.plan'.format( engine_name, cuda_idx  )
    engine_full_path = os.path.join( engine_path, engine_name )

    eng.save_engine(engine, engine_full_path) 



if __name__ == '__main__':
    cuda_idx =1




    engine_name = "resnet50"
    onnx_path = "resnet50.onnx"
    batch_size = 64

    for i in range(2):
        ebuilder( onnx_path, engine_path='', engine_name=engine_name, batch_size=batch_size, cuda_idx=i )
