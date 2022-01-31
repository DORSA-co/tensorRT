from importlib.resources import path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
from keras2onnx import convert_keras

K.set_learning_phase(0)




def keras_to_pb(model, output_filename, output_node_names):

   """
   This is the function to convert the Keras model to pb.

   Args:
      model: The Keras model.
      output_filename: The output .pb file name.
      output_node_names: The output nodes of the network. If None, then
      the function gets the last layer name as the output node.
   """
   paths = os.path.split(output_filename)
   paths = paths[:-1] #last one is the file name
   for i in range(len(paths)):
       __path__ = '/'.join(paths[:i+1])
       if not os.path.exists(__path__):
           os.mkdir(__path__)
   
   

   # Get the names of the input and output nodes.
   in_name = model.layers[0].get_output_at(0).name.split(':')[0]

   if output_node_names is None:
       output_node_names = [model.layers[-1].get_output_at(0).name.split(':')[0]]

   #sess = tf.compat.v1.keras.backend.get_session()
   full_model = tf.function(lambda x: model(x))
   full_model = full_model.get_concrete_function(
                tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

   # The TensorFlow freeze_graph expects a comma-separated string of output node names.
   frozen_graph_def = convert_variables_to_constants_v2(full_model)
   frozen_graph_def.graph.as_graph_def()
   
   

   wkdir = ''
   
   
   tf.io.write_graph(frozen_graph_def.graph,
                        wkdir,
                        output_filename,
                        as_text=False)

   return in_name, output_node_names



def keras_to_pb2(model, output_filename):
    
    paths = os.path.split(output_filename)
    for i in range(len(paths)):
       __path__ = '/'.join(paths[:i+1])
       if not os.path.exists(__path__):
           os.mkdir(__path__)
   
    
    model.save(output_filename)
    
    
def keras_to_onnx(model, output_filename): #for .hdf5
   #tf.saved_model.save(model, "tmp_model")
   #model = keras.models.load_model('tmp_model')
   onnx = convert_keras(model, output_filename)
   with open(output_filename, "wb") as f:
       f.write(onnx.SerializeToString())


if __name__=='__main__':
    

    model = keras.applications.resnet.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


    #in_tensor_name, out_tensor_names = keras_to_pb_nvidia(model, "models2/resnet50.pb", None) 
    keras_to_pb2(model, 'model/')
    
#    from keras2onnx import convert_keras
    
#    keras_to_onnx(model, 'test.onnx')