This Readme is for TensorFlow 1. 
For Tensorflow 2 users, please download the TF2 sample code provided in the dev blog.

# Requirements
Pre-install TensorRT 8 (tested on 8.0.0.3).

TF, Keras, [segmentation_model](https://github.com/qubvel/segmentation_models), [tf2onnx](https://github.com/onnx/tensorflow-onnx): 
```
pip install tensorflow-gpu==1.15.0 keras==2.3.1
pip install -U tf2onnx==1.8.2 pillow pycuda scikit-image segmentation-models
```

Download `cityscapes/labels.py` from the following link and put it in the same folder as the rest of the scripts:
  https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

Note: if you had a different version of TF or Keras, and you're still getting an error, run the following code and try re-installing all the requirements above.
```
pip uninstall keras keras-nightly keras-Preprocessing keras-vis tensorflow tensorflow-gpu -y
```

# ResNet50 (`TF-to-ONNX`)

## load ResNet50

```
python loadResNet.py
```
This will create `resnet50.pb`. The next step is to convert this file to onnx.

Go to the directory where you installed tf2onnx and run the following:
```
python -m tf2onnx.convert  --input /Path/to/resnet50.pb --inputs input_1:0 --outputs probs/Softmax:0 --output resnet50.onnx
```

## Build Engine:

`buildEngine.py` creates an engine from the onnx file. 

It accepts two inputs.

| Parameter     | Information                    | 
| ------------- | ------------------------------ |
| `--onnx_file` | Path to your onnx file         |
| `--plan_file` | This is the output engine file |

```
python buildEngine.py --onnx_file path/to/resnet50.onnx --plan_file resnet50.plan
```

# Semantic Segmentation
## Load and Save into  ONNX file

You need to download the `semantic_segmentation.hdf5` from the [link provided in the devblog](https://developer.download.nvidia.com/devblogs/semantic_segmentation.hdf5).
```
python loadSemanticSegmentation.py  --hdf5_file path/to/semantic_segmentation.hdf5 
``` 
This will create the `semantic_segmentation.onnx` file.

## Build Engine

```
python buildEngine.py --onnx_file path/to/semantic_segmentation.onnx --plan_file semantic.plan
```
This will create the `semantic.plan` file that can will be used later.

## Inference 

Please make sure you download `labels.py` from the Cityscapes github that we provided in the dev blog.
You will also need to download a sample image from the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) to serve as input for the semantic segmentation model.

```
python inference_semantic_seg.py --input_image path/to/input_image --engine_file path/to/semantic.plan --hdf5_file path/to/semantic_segmentation.hdf5 --height height_of_input --width width_of input
```

This will create two images `trt_output.png` and `keras_output.png`. you can compare them. 

# Unet
```
python unet_example.py
```
