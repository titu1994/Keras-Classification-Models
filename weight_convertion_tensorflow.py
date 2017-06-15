import os
import numpy as np

from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model

''' IMPORT YOUR SCRIPT FILE HERE TO CREATE YOUR MODEL LATER '''


''' BACKEND must be TENSORFLOW

This is a script to convert Tensorflow models (Tensorflow Backend, TF dim ordering)
to the other possible backend / dim ordering combinations.

Given weights and model for TF-kernels-TF-dim-ordering, produces a folder with
- TH-kernels-TF-dim-ordering
- TF-kernels-TH-dim-ordering
- TH-kernels-TH-dim-ordering

Needs 3 important inputs:

1) Theano model (model with TH dim ordering)
2) Tensorflow model (model with TF dim ordering)
3) Weight file for Theano model (tensorflow-kernels-tf-dim-ordering)

Supports : Multiple weights for same model (auto converts different weights for same model)

Usage:
1) Place script in the same directory as the weight file directory. If you want to place somewhere
   else, then you must provide absolute path to the weight files below instead of relative paths.

2) Edit the script to create your model :
    a) Import your model building script above (in the imports section)
    b) Set `th_dim_model` = ... (create your th dim model here and set it to th_dim_model)
    c) Set `tf_dim_model` = ... (create your tf dim model here and set it to tf_dim_model)
    d) Add the path to the weight files in `model_weights`.
       Note : The weight files must be for the Tensorflow model (tensorflow kernels, tf dim ordering)

3) Run the script.

4) Use the weight files in the created folders : ["th-kernels-th-dim/", "tf-kernels-th-dim/", "th-kernels-tf-dim/"]
'''

K.set_image_data_format('channels_first')
th_dim_model = None # Create your theano model here with TH dim ordering

K.set_image_data_format('channels_last')
tf_dim_model = None # Create your tensorflow model with TF dimordering here

model_weights = [''] # Add names of tensorflow model weight file paths here.
                     # These weights are assumed to be for tensorflow backend
                     # (tf kernels) with tf dim ordering!

"""

No need to edit anything below this. Simply run the script now after
editing the above 3 inputs.

"""


def shuffle_rows(original_w, nb_last_conv, nb_rows_dense):
    ''' Note :
    This algorithm to shuffle dense layer rows was provided by Kent Sommers (@kentsommer)
    in a gist : https://gist.github.com/kentsommer/e872f65926f1a607b94c2b464a63d0d3
    '''
    converted_w = np.zeros(original_w.shape)
    count = 0
    for index in range(original_w.shape[0]):
        if (index % nb_last_conv) == 0 and index != 0:
            count += 1
        new_index = ((index % nb_last_conv) * nb_rows_dense) + count
        print("index from " + str(index) + " -> " + str(new_index))
        converted_w[index] = original_w[new_index]

    return converted_w


first_dense = True
nb_last_conv = 0

for dirpath in ["th-kernels-channels-first-dim-ordering/",
                "tf-kernels-channels-first-dim-ordering/",
                "th-kernels-channels-last-dim-ordering/"]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# Converts (tensorflow kernels, tf dim ordering) to (theano kernels, tf dim ordering)
K.set_image_dim_ordering('th')
for weight_fn in model_weights:
    tf_dim_model.load_weights(weight_fn)
    convert_all_kernels_in_model(tf_dim_model)

    tf_dim_model.save_weights("th-kernels-channels-last-dim-ordering/%s" % weight_fn, overwrite=True)
    print("Done th-kernels-channels-last-dim-ordering %s" % weight_fn)


# Converts (tensorflow kernels, tf dim ordering) to (theano kernels, th dim ordering)
K.set_image_dim_ordering('tf')
for weight_fn in model_weights:
    tf_dim_model.load_weights(weight_fn) # tf-kernels-tf-dim
    convert_all_kernels_in_model(tf_dim_model) # th-kernels-tf-dim

    count_dense = 0
    for layer in tf_dim_model.layers:
        if layer.__class__.__name__ == "Dense":
            count_dense += 1

    if count_dense == 1:
        first_dense = False # If there is only 1 dense, no need to perform row shuffle in Dense layer

    print("Nb layers : ", len(tf_dim_model.layers))

    for index, tf_layer in enumerate(tf_dim_model.layers):
        if tf_layer.__class__.__name__ in ['Conv1D',
                                           'Conv2D',
                                           'Conv3D',
                                           'AtrousConvolution1D'
                                           'AtrousConvolution2D',
                                           'Conv2DTranspose',
                                           'SeparableConv2D',
                                           'DepthwiseConv2D',
                                           ]:
            weights = tf_layer.get_weights() # th-kernels-tf-dim
            weights[0] = weights[0].transpose((3, 2, 0, 1))
            th_dim_model.layers[index].set_weights(weights) # th-kernels-tf-dim

            nb_last_conv = tf_layer.nb_filter # preserve last number of convolutions to use with dense layers
            print("Converted layer %d : %s" % (index + 1, tf_layer.name))
        else:
            if tf_layer.__class__.__name__ == "Dense" and first_dense:
                weights = tf_layer.get_weights()
                nb_rows_dense_layer = weights[0].shape[0] // nb_last_conv

                print("Magic Number 1 : ", nb_last_conv)
                print("Magic nunber 2 : ", nb_rows_dense_layer)

                weights[0] = shuffle_rows(weights[0], nb_last_conv, nb_rows_dense_layer)
                th_dim_model.layers[index].set_weights(weights)

                first_dense = False
                print("Shuffled Dense Weights layer and saved %d : %s" % (index + 1, tf_layer.name))
            else:
                th_dim_model.layers[index].set_weights(tf_layer.get_weights())
                print("Saved layer %d : %s" % (index + 1, tf_layer.name))


    th_dim_model.save_weights("th-kernels-channels-first-dim-ordering/%s" % weight_fn, overwrite=True)
    print("Done th-kernels-channels-first-dim-ordering %s" % weight_fn)


# Converts (tensorflow kernels, tf dim ordering) to (tensorflow kernels, th dim ordering)
for weight_fn in model_weights:
    tf_dim_model.load_weights(weight_fn)

    for index, tf_layer in enumerate(tf_dim_model.layers):
        if tf_layer.__class__.__name__ in ['Conv1D',
                                           'Conv2D',
                                           'Conv3D',
                                           'AtrousConvolution1D'
                                           'AtrousConvolution2D',
                                           'Conv2DTranspose',
                                           'SeparableConv2D',
                                           'DepthwiseConv2D',
                                           ]:
            weights = tf_layer.get_weights()
            weights[0] = weights[0].transpose((3, 2, 0, 1))
            th_dim_model.layers[index].set_weights(weights)
        else:
            th_dim_model.layers[index].set_weights(tf_layer.get_weights())

        print("Changed dim %d : %s" % (index + 1, tf_layer.name))

    th_dim_model.save_weights("tf-kernels-channels-first-dim-ordering/%s" % weight_fn, overwrite=True)
    print("Done tf-kernels-channels-first-dim-ordering %s" % weight_fn)