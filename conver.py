from tflite import Model
import numpy as np
from collections import OrderedDict
from facial_lm_model import FacialLM_Model
from utils import GetKeysDict
import torch

data = open("model_weights/face_landmark.tflite", "rb").read()
model = Model.GetRootAsModel(data, 0)

tflite_graph = model.Subgraphs(0)
tflite_graph.Name()

# Tensor name to index mapping
tflite_tensor_dict = {}
for i in range(tflite_graph.TensorsLength()):
    tflite_tensor_dict[tflite_graph.Tensors(i).Name().decode("utf8")] = i 

def get_weights(tensor_name):
    index = tflite_tensor_dict[tensor_name]
    tensor = tflite_graph.Tensors(index)

    buffer = tensor.Buffer()
    shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]

    weights = model.Buffers(buffer).DataAsNumpy()
    weights = weights.view(dtype=np.float32)
    weights = weights.reshape(shape)
    return weights


# Store the weights in dict
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

pt_model = FacialLM_Model()
# pt_model(torch.randn(2,3,64,64))[0].shape

probable_names = []
for i in range(0, tflite_graph.TensorsLength()):
    tensor = tflite_graph.Tensors(i)
    if tensor.Buffer() > 0 and tensor.Type() == 0:
        probable_names.append(tensor.Name().decode("utf-8"))

pt2tflite_keys = {}
i = 0
for name, params in pt_model.state_dict().items():
    # first 83 nodes names are perfectly matched
    if i < 83:
        pt2tflite_keys[name] = probable_names[i]
        i += 1

# Remaining nodes
matched_keys = GetKeysDict().facial_landmark_dict

# update the remaining keys
pt2tflite_keys.update(matched_keys)

new_state_dict = OrderedDict()

for pt_key, tflite_key in pt2tflite_keys.items():
    weight = get_weights(tflite_key)
    print(pt_key, tflite_key, weight.shape, pt_model.state_dict()[pt_key].shape)

    # if pt_key == 'facial_landmarks.4.weight':
        # weight = weight.transpose((0, 3, 1, 2)) 
        # weight = weight.transpose((0, 3, 2, 1)) 
        # print(weight.shape)
        # print(weight)
    # else:
    if weight.ndim == 4:
        if 'depthwise' in tflite_key:
            # (1, 3, 3, 32) --> (32, 1, 3, 3)
            # for depthwise conv
            weight = weight.transpose((3, 0, 1, 2))  
        else:
            weight = weight.transpose((0, 3, 1, 2)) 

    if 'p_re_lu' in tflite_key:
        weight = weight.reshape(-1)
    
    new_state_dict[pt_key] = torch.from_numpy(weight)

pt_model.load_state_dict(new_state_dict, strict=True)

torch.save(pt_model.state_dict(), "facial_landmarks.pth")