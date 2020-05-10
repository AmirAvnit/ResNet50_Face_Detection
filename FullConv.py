"""
Adapted from: https://github.com/gabrieldemarmiesse/heatmaps
Modulated by Amir Avnit
"""

from keras.preprocessing import image
from keras.layers import *
from keras.applications.resnet50 import preprocess_input
from keras.models import Model, Sequential
import matplotlib.pyplot as plt

def layer_type(layer):
    return str(layer)[10:].split(" ")[0].split(".")[-1]

def detect_configuration(model):

    # Get names (i.e., types) of layers from end to beggining
    inverted_list_layers = [layer_type(layer) for layer in model.layers[::-1]]

    layer1 = None
    layer2 = None

    i = len(model.layers)

    for layer in inverted_list_layers:
        i -= 1
        if layer2 is None:
            if layer == "GlobalAveragePooling2D" or layer == "GlobalMaxPooling2D":
                layer2 = layer

            elif layer == "Flatten":
                return "local pooling - flatten", i - 1

        else:
            layer1 = layer
            break

    if layer1 == "MaxPooling2D" and layer2 == "GlobalMaxPooling2D":
        return i
    elif layer1 == "AveragePooling2D" and layer2 == "GlobalAveragePooling2D":
        return i

    elif layer1 == "MaxPooling2D" and layer2 == "GlobalAveragePooling2D":
        return i + 1
    elif layer1 == "AveragePooling2D" and layer2 == "GlobalMaxPooling2D":
        return i + 1

    else: # (global pooling)
        return i + 1

def convert_to_functional(model):
    
    input_tensor = Input(batch_shape=K.int_shape(model.input))
    x = input_tensor
    
    for layer in model.layers:
        x = layer(x)
    
    return Model(input_tensor, x)


def insert_weights(layer, new_layer):
    W, b = layer.get_weights()
    ax1, ax2, previous_filter, n_filter = new_layer.get_weights()[0].shape
    new_W = W.reshape((ax1, ax2, previous_filter, n_filter))
    new_W = new_W.transpose((0, 1, 2, 3))

    new_layer.set_weights([new_W, b])
    
def copy_last_layers(model, begin, x, last_activation='linear', activation='softmax'):
    i = begin

    for layer in model.layers[begin:]:
        if layer_type(layer) == "Dense":
            last_activation = layer.get_config()["activation"]
            if i == len(model.layers) - 1:
                x = add_reshaped_layer(layer, x, 1, no_activation=True)
            else:
                x = add_reshaped_layer(layer, x, 1)

        elif layer_type(layer) == "Dropout" or layer_type(layer) == 'Reshape':
            pass

        elif layer_type(layer) == "Activation" and i == len(model.layers) - 1:
            last_activation = layer.get_config()['activation']
            break
        else:
            x = add_to_model(x, layer)
        i += 1

        x = Activation(activation)(x)
    return x

def add_reshaped_layer(layer, x, size, no_activation=False, atrous_rate=None):
    conf = layer.get_config()

    if no_activation:
        activation = "linear"
    else:
        activation = conf["activation"]

    if size == 1:
        new_layer = Conv2D(conf["units"], (size, size),
                           activation=activation, name=conf['name'])
    else:
        new_layer = Conv2D(conf["units"], (size, size),
                           dilation_rate=(atrous_rate, atrous_rate),
                           activation=activation, padding='valid',
                           name=conf['name'])

    x = new_layer(x)
    # We transfer the weights:
    insert_weights(layer, new_layer)
    return x

def display_heatmap(model, img_path, ids, preprocessing=None, target_size= (1200,1200)):

    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    if preprocessing is not None:
        x = preprocess_input(x)

    out = model.predict(x)

    heatmap = out[0]  # Remove batch axis
   
    heatmap = heatmap[:, :, ids]
    if heatmap.ndim == 3:
        heatmap = np.sum(heatmap, axis=2)

    plt.imshow(heatmap, interpolation="none", cmap='seismic')
    plt.show()

    
def to_fullconv(model, input_shape=None):
    
    if isinstance(model, Sequential):
        model = convert_to_functional(model)

    index = detect_configuration(model)

    img_input = Input(shape=(None, None, 3))
    
    # Part of the model to keep 
    middle_model = Model(inputs=model.input, outputs=model.layers[index - 1].get_output_at(-1))

    x = middle_model(img_input)

    print("Model cut at layer: " + str(index))

    x = copy_last_layers(model, index + 1, x)

    return Model(img_input, x)

