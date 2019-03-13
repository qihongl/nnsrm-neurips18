# reference: https://github.com/philipperemy/keras-visualize-activations
import keras.backend as K
import numpy as np


def get_activations(model, model_inputs, testing_mode=True,
                    print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out])
             for out in outputs]  # evaluation functions

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    mode_flag = 0 if testing_mode else 1
    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(mode_flag)
    else:
        list_inputs = [model_inputs, mode_flag]

    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            sp = layer_activations.shape
            print('%s - %d units' % (str(sp), round(np.cumprod(sp)[-1]/sp[0])))
        else:
            print(layer_activations)
    return activations

#
# def display_activations(activation_maps):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     """
#     (1, 26, 26, 32)
#     (1, 24, 24, 64)
#     (1, 12, 12, 64)
#     (1, 12, 12, 64)
#     (1, 9216)
#     (1, 128)
#     (1, 128)
#     (1, 10)
#     """
#     batch_size = activation_maps[0].shape[0]
#     assert batch_size == 1, 'One image at a time to visualize.'
#     for i, activation_map in enumerate(activation_maps):
#         print('Displaying activation map {}'.format(i))
#         shape = activation_map.shape
#         if len(shape) == 4:
#             activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
#         elif len(shape) == 2:
#             # try to make it square as much as possible. we can skip some activations.
#             activations = activation_map[0]
#             num_activations = len(activations)
#             if num_activations > 1024:  # too hard to display it on the screen.
#                 square_param = int(np.floor(np.sqrt(num_activations)))
#                 activations = activations[0: square_param * square_param]
#                 activations = np.reshape(activations, (square_param, square_param))
#             else:
#                 activations = np.expand_dims(activations, axis=0)
#         else:
#             raise Exception('len(shape) = 3 has not been implemented.')
#         plt.imshow(activations, interpolation='None', cmap='jet')
#         plt.show()
