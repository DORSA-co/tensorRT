import os

os.system("ls")



def pb2onnx(input_path, ouput_path):
    command = 'python -m tf2onnx.convert --saved-model {} --output {}'.format(
        input_path,
        ouput_path
    )
    os.system(command)
    


if __name__ == '__main__':
    #pb2onnx('models/resnet50.pb', 'resnet50.onnx')
    pb2onnx('model/', 'resnet50.onnx')
    