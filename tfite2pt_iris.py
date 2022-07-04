from tflite import Model
import numpy as np
from collections import OrderedDict

from blazeiris import IrisLM,GetKeysDict
import torch



data = open("model_weights\\iris_landmark.tflite", "rb").read()
model = Model.GetRootAsModel(data, 0)

tflite_graph = model.Subgraphs(0)
tflite_graph.Name()

tflite_tensor_dict = {}
for i in range(tflite_graph.TensorsLength()):
    tflite_tensor_dict[tflite_graph.Tensors(i).Name().decode("utf8")] = i 

parameters = {}
for i in range(tflite_graph.TensorsLength()):
    tensor = tflite_graph.Tensors(i)
    if tensor.Buffer() > 0:
        name = tensor.Name().decode("utf8")
        parameters[name] = tensor.Buffer()
    else:
        # Buffer value less than zero are not weights
        print(tensor.Name().decode("utf8"))

print("Total parameters: ", len(parameters))


def get_weights(tensor_name):
    index = tflite_tensor_dict[tensor_name]
    tensor = tflite_graph.Tensors(index)

    buffer = tensor.Buffer()
    shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]

    weights = model.Buffers(buffer).DataAsNumpy()
    weights = weights.view(dtype=np.float32)
    weights = weights.reshape(shape)
    return weights


net = IrisLM()
# net(torch.randn(2,3,64,64))[0].shape

probable_names = []
for i in range(0, tflite_graph.TensorsLength()):
    tensor = tflite_graph.Tensors(i)
    if tensor.Buffer() > 0 and tensor.Type() == 0:
        probable_names.append(tensor.Name().decode("utf-8"))

pt2tflite_keys = {}
i = 0
for name, params in net.state_dict().items():
    print(name)
    if i < 85:
        pt2tflite_keys[name] = probable_names[i]
        i += 1

matched_keys = GetKeysDict().iris_landmark_dict 
pt2tflite_keys.update(matched_keys)

new_state_dict = OrderedDict()

for pt_key, tflite_key in pt2tflite_keys.items():
    weight = get_weights(tflite_key)
    print(pt_key, tflite_key)

    if weight.ndim == 4:
        if weight.shape[0] == 1: 
            weight = weight.transpose((3, 0, 1, 2))  
        else:
            weight = weight.transpose((0, 3, 1, 2)) 
    elif weight.ndim == 3:
        weight = weight.reshape(-1)
    
    new_state_dict[pt_key] = torch.from_numpy(weight)

net.load_state_dict(new_state_dict, strict=True)

torch.save(net.state_dict(), "model_weights/irislandmarks.pth")