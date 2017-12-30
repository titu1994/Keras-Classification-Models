# Keras-Classification-Models
A set of models which allow easy creation of Keras models to be used for classification purposes. Also contains
modules which offer implementations of recent papers.

# <a href="https://github.com/titu1994/neural-architecture-search">Neural Architecture Search in Keras</a>
Basic implementation of Controller RNN from the paper ["Neural Architecture Search with Reinforcement Learning
"](https://arxiv.org/abs/1611.01578) and ["Learning Transferable Architectures for Scalable Image Recognition"](https://arxiv.org/abs/1707.07012).

- Uses Keras to define and train children / generated networks, which are defined in Tensorflow by the Controller RNN.
- Define a state space by using StateSpace, a manager which adds states and handles communication between the Controller RNN and the user.
- Reinforce manages the training and evaluation of the Controller RNN
- NetworkManager handles the training and reward computation of a Keras model

Available at : <a href="https://github.com/titu1994/neural-architecture-search">Neural Architecture Search in Keras</a>

# <a href="https://github.com/titu1994/keras-non-local-nets">Non-Local Neural Networks in Keras</a>
Keras implementation of Non-local blocks from the paper ["Non-local Neural Networks"](https://arxiv.org/abs/1711.07971).

- Support for "Gaussian", "Embedded Gaussian" and "Dot" instantiations of the Non-Local block.
- Support for shielded computation mode (reduces computation by 4x)
- Support for "Concatenation" instantiation will be supported when authors release their code.

Available at : <a href="https://github.com/titu1994/keras-non-local-nets">Non-Local Neural Networks in Keras</a>

#  <a href="https://github.com/titu1994/Keras-NASNet">Neural Architecture Search Net (NASNet) in Keras</a>
An implementation of "NASNet" models from the paper [Learning Transferable Architectures for Scalable Image Recognitio](https://arxiv.org/abs/1707.07012) in Keras 2.0+.

Supports building NASNet Large (6 @ 4032), NASNet Mobile (4 @ 1056) and custom NASNets. 

Available at : <a href="https://github.com/titu1994/Keras-NASNet">Neural Architecture Search Net (NASNet) in Keras</a>

#  <a href="https://github.com/titu1994/keras-squeeze-excite-network">Squeeze and Excite Networks in Keras</a>
Implementation of Squeeze and Excite networks in Keras. Supports ResNet and Inception v3 models currently. Support for Inception v4 and Inception-ResNet-v2 will also come once the paper comes out.

Available at : <a href="https://github.com/titu1994/keras-squeeze-excite-network">Squeeze and Excite Networks in Keras</a>

# <a href="https://github.com/titu1994/Keras-DualPathNetworks">Dual Path Networks in Keras</a>
Implementation of [Dual Path Networks](https://arxiv.org/abs/1707.01629), which combine the grouped convolutions of ResNeXt with the dense connections of DenseNet into two path

Available at : <a href="https://github.com/titu1994/MobileNetworks">Dual Path Networks in Keras</a>

# <a href="https://github.com/titu1994/MobileNetworks">MobileNets in Keras</a>
Implementation of MobileNet models from the paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf) in Keras 2.0+.

Contains code for building the MobileNet model (optimized for datasets similar to ImageNet) and weights for the model trained on ImageNet.

Available at : <a href="https://github.com/titu1994/MobileNetworks">MobileNets in Keras</a>

# <a href="https://github.com/titu1994/Keras-ResNeXt">ResNeXt in Keras</a>
Implementation of ResNeXt models from the paper [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf) in Keras 2.0+.

Contains code for building the general ResNeXt model (optimized for datasets similar to CIFAR) and ResNeXtImageNet (optimized for the ImageNet dataset).

Available at : <a href="https://github.com/titu1994/Keras-ResNeXt">ResNeXt in Keras</a>

-----

# <a href="https://github.com/titu1994/Inception-v4">Inception v4 in Keras</a>
Implementations of the Inception-v4, Inception - Resnet-v1 and v2 Architectures in Keras using the Functional API. 
The paper on these architectures is available at <a href="https://arxiv.org/pdf/1602.07261v1.pdf">"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning".</a>

The models are plotted and shown in the architecture sub folder. 
Due to lack of suitable training data (ILSVR 2015 dataset) and limited GPU processing power, the weights are not provided.

Contains : Inception v4, Inception-ResNet-v1 and Inception-ResNet-v2

Available at : <a href="https://github.com/titu1994/Inception-v4">Inception v4 in Keras </a>

-----

# <a href="https://github.com/titu1994/Wide-Residual-Networks">Wide Residual Networks in Keras</a>
Implementation of Wide Residual Networks from the paper <a href="https://arxiv.org/pdf/1605.07146v1.pdf">Wide Residual Networks</a>

## Usage
It can be used by importing the wide_residial_network script and using the create_wide_residual_network() method. There are several parameters which can be changed to increase the depth or width of the network.

Note that the number of layers can be calculated by the formula : nb_layers = 4 + 6 * N 

```
import wide_residial_network as wrn
ip = Input(shape=(3, 32, 32)) # For CIFAR 10

wrn_28_10 = wrn.create_wide_residual_network(ip, nb_classes=10, N=4, k=10, dropout=0.0, verbose=1)

model = Model(ip, wrn_28_10)
```

Contains weights for WRN-16-8 and WRN-28-8 models trained on the CIFAR-10 Dataset.

Available at : <a href="https://github.com/titu1994/Wide-Residual-Networks">Wide Residual Network in Keras</a>

------

# <a href="https://github.com/titu1994/DenseNet">DenseNet in Keras</a>
Implementation of DenseNet from the paper [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993v1.pdf).

## Usage

1. Run the cifar10.py script to train the DenseNet 40 model
2. Comment out the model.fit_generator(...) line and uncomment the model.load_weights("weights/DenseNet-40-12-CIFAR10.h5") line to test the classification accuracy.

Contains weights for DenseNet-40-12 and DenseNet-Fast-40-12, trained on CIFAR 10.

Available at : <a href="https://github.com/titu1994/DenseNet">DenseNet in Keras</a>

------

# <a href='https://github.com/titu1994/Snapshot-Ensembles'>Snapshot Ensembles in Keras</a>
Implementation of the paper [Snapshot Ensembles](https://github.com/titu1994/Snapshot-Ensembles)

## Usage
The technique is simple to implement in Keras, using a custom callback. These callbacks can be built using the SnapshotCallbackBuilder class in snapshot.py. Other models can simply use this callback builder to other models to train them in a similar manner.

1. Download the 6 WRN-16-4 weights that are provided in the Release tab of the project and place them in the weights directory
2. Run the train_cifar_10.py script to train the WRN-16-4 model on CIFAR-10 dataset (not required since weights are provided)
3. Run the predict_cifar_10.py script to make an ensemble prediction.

Contains weights for WRN-CIFAR100-16-4 and WRN-CIFAR10-16-4 (snapshot ensemble weights - ranging from 1-5 and including single best model)

Available at : <a href='https://github.com/titu1994/Snapshot-Ensembles'>Snapshot Ensembles in Keras</a>

-----

# <a href="https://github.com/titu1994/Residual-of-Residual-Networks">Residual Networks of Residual Networks in Keras</a>
Implementation of the paper ["Residual Networks of Residual Networks: Multilevel Residual Networks"](https://arxiv.org/pdf/1608.02908v1.pdf)

## Usage
To create RoR ResNet models, use the `ror.py` script :
```
import ror

input_dim = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)
model = ror.create_residual_of_residual(input_dim, nb_classes=100, N=2, dropout=0.0) # creates RoR-3-110 (ResNet)
```

To create RoR Wide Residual Network models, use the `ror_wrn.py` script :
```
import ror_wrn as ror

input_dim = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)
model = ror.create_pre_residual_of_residual(input_dim, nb_classes=100, N=6, k=2, dropout=0.0) # creates RoR-3-WRN-40-2 (WRN)
```
Contains weights for RoR-3-WRN-40-2 trained on CIFAR 10

Available at : <a href="https://github.com/titu1994/Residual-of-Residual-Networks">Residual Networks of Residual Networks in Keras</a>

-----

# Keras Segmentation Models

A set of models which allow easy creation of Keras models to be used for segmentation tasks.

# <a href="https://github.com/titu1994/Fully-Connected-DenseNets-Semantic-Segmentation">Fully Connected DenseNets for Semantic Segmentation</a>
Implementation of the paper [The One Hundred Layers Tiramisu : Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326v1.pdf)

## Usage

Simply import the densenet_fc.py script and call the create method:
```
import densenet_fc as dc

model = dc.create_fc_dense_net(img_dim=(3, 224, 224), nb_dense_block=5, growth_rate=12,
                               nb_filter=16, nb_layers=4)
```
-----

# Keras Modules

A set of scripts which can be used to add advanced functionality to Keras.

# <a href="https://github.com/titu1994/keras-SRU">Simple Recurrent Unit (SRU)</a>
Implementation of the paper [Training RNNs as Fast as CNNs](https://arxiv.org/abs/1709.02755) for Keras 2.0+. SRU is a recurrent unit that can run over 10 times faster than cuDNN LSTM, without loss of accuracy tested on many tasks, when implemented with a custom CUDA kernel.

This is a naive implementation with some speed gains over the generic LSTM cells, however its speed is not yet 10x that of cuDNN LSTMs.

-----

# <a href="https://github.com/titu1994/BatchRenormalization">Batch Renormalization</a>
Batch Renormalization algorithm implementation in Keras 1.2.1. Original paper by Sergey Ioffe, [Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models](https://arxiv.org/pdf/1702.03275.pdf).\

## Usage

Add the `batch_renorm.py` script into your repository, and import the BatchRenormalization layer.

Eg. You can replace Keras BatchNormalization layers with BatchRenormalization layers.
```
from batch_renorm import BatchRenormalization
```
-----

# <a href="https://github.com/titu1994/Keras-Multiplicative-LSTM">Multiplicative LSTM</a>
Implementation of the paper [Multiplicative LSTM for sequence modelling](https://arxiv.org/pdf/1609.07959.pdf) for Keras 2.0+. Multiplicative LSTMs have been shown to achieve state-of-the-art or close to SotA results for sequence modelling datasets. They also perform better than stacked LSTM models for the Hutter-prize dataset and the raw wikipedia dataset.


## Usage

Add the `multiplicative_lstm.py` script into your repository, and import the MultiplicativeLSTM layer.

Eg. You can replace Keras LSTM layers with MultiplicativeLSTM layers.
```
from multiplicative_lstm import MultiplicativeLSTM
```
-----

