# Keras-Classification-Models
A set of models which allow easy creation of Keras models to be used for classification purposes. Also contains
modules which offer implementations of recent papers.

# **NOTE**
Since this readme is getting very large, I will post most of these projects on [titu1994.github.io](http://titu1994.github.io)

# Image Classification Models

# <a href="https://github.com/titu1994/keras-octconv">Keras Octave Convolutions</a>

Keras implementation of the Octave Convolution blocks from the paper [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049).

<img src="https://github.com/titu1994/keras-octconv/raw/master/images/octconv.png?raw=true" height=100% width=100%>

-----

#  <a href="https://github.com/titu1994/keras-SparseNet">Sparse Neural Networks (SparseNets) in Keras</a>
An implementation of "SparseNets" from the paper [Sparsely Connected Convolutional Networks](https://arxiv.org/abs/1801.05895) in Keras 2.0+.

SparseNets are a modification of DenseNet and its dense connectivity pattern to reduce memory requirements drastically while still having similar or better performance.

-----

# <a href="https://github.com/titu1994/keras-non-local-nets">Non-Local Neural Networks in Keras</a>
Keras implementation of Non-local blocks from the paper ["Non-local Neural Networks"](https://arxiv.org/abs/1711.07971).

- Support for "Gaussian", "Embedded Gaussian" and "Dot" instantiations of the Non-Local block.
- Support for shielded computation mode (reduces computation by 4x)
- Support for "Concatenation" instantiation will be supported when authors release their code.

Available at : <a href="https://github.com/titu1994/keras-non-local-nets">Non-Local Neural Networks in Keras</a>

-----

#  <a href="https://github.com/titu1994/Keras-NASNet">Neural Architecture Search Net (NASNet) in Keras</a>
An implementation of "NASNet" models from the paper [Learning Transferable Architectures for Scalable Image Recognitio](https://arxiv.org/abs/1707.07012) in Keras 2.0+.

Supports building NASNet Large (6 @ 4032), NASNet Mobile (4 @ 1056) and custom NASNets. 

Available at : <a href="https://github.com/titu1994/Keras-NASNet">Neural Architecture Search Net (NASNet) in Keras</a>

-----

#  <a href="https://github.com/titu1994/keras-squeeze-excite-network">Squeeze and Excite Networks in Keras</a>
Implementation of Squeeze and Excite networks in Keras. Supports ResNet and Inception v3 models currently. Support for Inception v4 and Inception-ResNet-v2 will also come once the paper comes out.

Available at : <a href="https://github.com/titu1994/keras-squeeze-excite-network">Squeeze and Excite Networks in Keras</a>

-----

# <a href="https://github.com/titu1994/Keras-DualPathNetworks">Dual Path Networks in Keras</a>
Implementation of [Dual Path Networks](https://arxiv.org/abs/1707.01629), which combine the grouped convolutions of ResNeXt with the dense connections of DenseNet into two path

Available at : <a href="https://github.com/titu1994/MobileNetworks">Dual Path Networks in Keras</a>

-----

# <a href="https://github.com/titu1994/MobileNetworks">MobileNets in Keras</a>
Implementation of MobileNet models from the paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf) in Keras 2.0+.

Contains code for building the MobileNet model (optimized for datasets similar to ImageNet) and weights for the model trained on ImageNet.

Also contains MobileNet V2 model implementations + weights.

Available at : <a href="https://github.com/titu1994/MobileNetworks">MobileNets in Keras</a>

-----

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

# Neural Architecture Search

# <a href="https://github.com/titu1994/pyshac">Sequentual Halving and Classification</a>

PySHAC is a python library to use the Sequential Halving and Classification algorithm from the paper [Parallel Architecture and Hyperparameter Search via Successive Halving and Classification](https://arxiv.org/abs/1805.10255) with ease.

Available at : <a href="https://github.com/titu1994/pyshac">Sequentual Halving and Classification</a>
Documentation available at : <a href="http://titu1994.github.io/pyshac/">PySHAC Documentation</a>

-----

# <a href="https://github.com/titu1994/progressive-neural-architecture-search">Progressive Neural Architecture Search in Keras</a>
Basic implementation of Encoder RNN from the paper ["Progressive Neural Architecture Search"]https://arxiv.org/abs/1712.00559), which is an improvement over the original Neural Architecture Search paper since it requires far less time and resources.

- Uses Keras to define and train children / generated networks, which are defined in Tensorflow by the Encoder RNN.
- Define a state space by using StateSpace, a manager which adds states and handles communication between the Encoder RNN and the user. Submit custom operations and parse locally as required.
- Encoder RNN trained using a modified Sequential Model Based Optimization algorithm from the paper. Some stability modifications made by me to prevent extreme variance when training to cause failed training.
- NetworkManager handles the training and reward computation of a Keras model

Available at : <a href="https://github.com/titu1994/progressive-neural-architecture-search">Progressive Neural Architecture Search in Keras</a>

-----

# <a href="https://github.com/titu1994/neural-architecture-search">Neural Architecture Search in Keras</a>
Basic implementation of Controller RNN from the paper ["Neural Architecture Search with Reinforcement Learning
"](https://arxiv.org/abs/1611.01578) and ["Learning Transferable Architectures for Scalable Image Recognition"](https://arxiv.org/abs/1707.07012).

- Uses Keras to define and train children / generated networks, which are defined in Tensorflow by the Controller RNN.
- Define a state space by using StateSpace, a manager which adds states and handles communication between the Controller RNN and the user.
- Reinforce manages the training and evaluation of the Controller RNN
- NetworkManager handles the training and reward computation of a Keras model

Available at : <a href="https://github.com/titu1994/neural-architecture-search">Neural Architecture Search in Keras</a>

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


# Keras Recurrent Neural Networks

A set of scripts which can be used to add custom Recurrent Neural Networks to Keras.

-----

# [Neural Algorithmic Logic Units](https://github.com/titu1994/keras-neural-alu)

A Keras implementation of Neural Arithmatic and Logical Unit from the paper [Neural Algorithmic Logic Units](https://arxiv.org/abs/1808.00508)
by Andrew Trask, Felix Hill, Scott Reed, Jack Rae, Chris Dyer, Phil Blunsom.

- Contains the layers for `Neural Arithmatic Logic Unit (NALU)` and `Neural Accumulator (NAC)`.
- Also contains the results of the static function learning toy tests.

-----

# <a href="https://github.com/titu1994/Keras-just-another-network-JANET">Chrono Initializer, Chrono LSTM and JANET</a>
Keras implementation of the paper [The unreasonable effectiveness of the forget gate](https://arxiv.org/abs/1804.04849) and the Chrono initializer and Chrono LSTM from the paper [Can Recurrent Neural Networks Warp Time?](https://openreview.net/pdf?id=SJcKhk-Ab). 

This model utilizes just 2 gates - forget (f) and context (c) gates out of the 4 gates in a regular LSTM RNN, and uses `Chrono Initialization` to acheive better performance than regular LSTMs while using fewer parameters and less complicated gating structure.

## Usage
Simply import the `janet.py` file into your repo and use the `JANET` layer. 

It is **not** adviseable to use the `JANETCell` directly wrapped around a `RNN` layer, as this will not allow the `max timesteps` calculation that is needed for proper training using the `Chrono Initializer` for the forget gate.

The `chrono_lstm.py` script contains the `ChronoLSTM` model, as it requires minimal modifications to the original `LSTM` layer to use the `ChronoInitializer` for the forget and input gates.

Same restrictions to usage as the `JANET` layer, use the `ChronoLSTM` layer directly instead of the `ChronoLSTMCell` wrapped around a `RNN` layer.

```python
from janet import JANET
from chrono_lstm import ChronoLSTM

...
```

To use just the `ChronoInitializer`, import the `chrono_initializer.py` script.

-----

# <a href="https://github.com/titu1994/Keras-IndRNN">Independently Recurrent Neural Networks (SRU)</a>
Implementation of the paper [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/abs/1803.04831) for Keras 2.0+. IndRNN is a recurrent unit that can run over extremely long time sequences, able to learn the additional problem over 5000 timesteps where most other models fail..

## Usage
Usage of IndRNNCells
```python
from ind_rnn import IndRNNCell, RNN

cells = [IndRNNCell(128), IndRNNCell(128)]
ip = Input(...)
x = RNN(cells)(ip)
...
```
Usage of IndRNN layer
```python
from ind_rnn import IndRNN

ip = Input(...)
x = IndRNN(128)(x)
...
```
-----

# <a href="https://github.com/titu1994/keras-SRU">Simple Recurrent Unit (SRU)</a>
Implementation of the paper [Training RNNs as Fast as CNNs](https://arxiv.org/abs/1709.02755) for Keras 2.0+. SRU is a recurrent unit that can run over 10 times faster than cuDNN LSTM, without loss of accuracy tested on many tasks, when implemented with a custom CUDA kernel.

This is a naive implementation with some speed gains over the generic LSTM cells, however its speed is not yet 10x that of cuDNN LSTMs.

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

# <a href="https://github.com/titu1994/keras-minimal-rnn">Minimal RNN</a>
Implementation of the paper [MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks
](https://arxiv.org/abs/1711.06788) for Keras 2.0+. Minimal RNNs are a new recurrent neural network architecture that achieves comparable performance as the popular gated RNNs with a simplified structure. It employs minimal updates within RNN, which not only leads to efficient learning and testing but more importantly better interpretability and trainability

## Usage

Import minimal_rnn.py and use either the MinimalRNNCell or MinimalRNN layer

```python
from minimal_rnn import MinimalRNN 

# this imports the layer rather than the cell
ip = Input(...)  # Rank 3 input shape
x = MinimalRNN(units=128)(ip)
...
```
-----

# <a href="https://github.com/titu1994/Nested-LSTM">Nested LSTM</a>
Implementation of the paper [Nested LSTMs](https://arxiv.org/abs/1801.10308) for Keras 2.0+. Nested LSTMs add depth to LSTMs via nesting as opposed to stacking. The value of a memory cell in an NLSTM is computed by an LSTM cell, which has its own inner memory cell. Nested LSTMs outperform both stacked and single-layer LSTMs with similar numbers of parameters in our experiments on various character-level language modeling tasks, and the inner memories of an LSTM learn longer term dependencies compared with the higher-level units of a stacked LSTM

## Usage
```python
from nested_lstm import NestedLSTM

ip = Input(shape=(nb_timesteps, input_dim))
x = NestedLSTM(units=64, depth=2)(ip)
...
```
-----

# Keras Modules

A set of scripts which can be used to add advanced functionality to Keras.

-----

# <a href="https://github.com/titu1994/keras-switchnorm">Switchable Normalization for Keras</a>
Switchable Normalization is a normalization technique that is able to learn different normalization operations for different normalization layers in a deep neural network in an end-to-end manner.

Keras port of the implementation of the paper Differentiable Learning-to-Normalize via Switchable Normalization.

Code ported from the switchnorm official repository.

# **Note**

This only implements the moving average version of batch normalization component from the paper. The batch average technique cannot be easily implemented in Keras as a layer, and therefore it is not supported.

## Usage
Simply import switchnorm.py and replace BatchNormalization layer with this layer.

```python
from switchnorm import SwitchNormalization

ip = Input(...)
...
x = SwitchNormalization(axis=-1)(x)
...
```

-----

# <a href="https://github.com/titu1994/Keras-Group-Normalization">Group Normalization for Keras</a>
A Keras implementation of [Group Normalization](https://arxiv.org/abs/1803.08494) by Yuxin Wu and Kaiming He.

Useful for fine-tuning of large models on smaller batch sizes than in research setting (where batch size is very large due to multiple GPUs). Similar to Batch Renormalization, but performs significantly better on ImageNet.

As can be seen, GN is independent of batchsize, which is crucial for fine-tuning large models which cannot be retrained with small batch sizes due to Batch Normalization's dependence on large batchsizes to compute the statistics of each batch and update its moving average perameters properly.

## Usage

Dropin replacement for BatchNormalization layers from Keras. The important parameter that is different from `BatchNormalization` is called `groups`. This must be appropriately set, and requires certain constraints such as :

1)  Needs to an integer by which the number of channels is divisible.
2)  `1 <= G <= #channels`, where #channels is the number of channels in the incomming layer.

```python
from group_norm import GroupNormalization

ip = Input(shape=(...))
x = GroupNormalization(groups=32, axis=-1)
...
```

-----

# <a href="https://github.com/titu1994/keras-normalized-optimizers">Normalized Optimizers for Keras</a>
Keras wrapper class for Normalized Gradient Descent from [kmkolasinski/max-normed-optimizer](https://github.com/kmkolasinski/deep-learning-notes/tree/master/max-normed-optimizer), which can be applied to almost all Keras optimizers.

Partially implements [Block-Normalized Gradient Method: An Empirical Study for Training Deep Neural Network](https://arxiv.org/abs/1707.04822) for all base Keras optimizers, and allows flexibility to choose any normalizing function. It does not implement adaptive learning rates however.

## Usage

```python
from keras.optimizers import Adam, SGD
from optimizer import NormalizedOptimizer

sgd = SGD(0.01, momentum=0.9, nesterov=True)
sgd = NormalizedOptimizer(sgd, normalization='l2')

adam = Adam(0.001)
adam = NormalizedOptimizer(adam, normalization='l2')
```

-----

# <a href="https://github.com/titu1994/tf-eager-examples">Tensorflow Eager with Keras APIs</a>
A set of example notebooks and scripts which detail the usage and pitfalls of Eager Execution Mode in Tensorflow using Keras high level APIs.

-----

# <a href="https://github.com/titu1994/keras-one-cycle">One Cycle Learning Rate Policy for Keras</a>
Implementation of One-Cycle Learning rate policy from the papers by Leslie N. Smith.

- [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)
- [Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)

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



