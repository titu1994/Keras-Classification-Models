# Keras-Classification-Models
A set of models which allow easy creation of Keras models to be used for classification purposes.

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
