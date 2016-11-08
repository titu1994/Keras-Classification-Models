# Keras-Classification-Models
A set of models which allow easy creation of Keras models to be used for classification purposes.

# <a href="https://github.com/titu1994/Inception-v4">Inception v4 in Keras</a>
Implementations of the Inception-v4, Inception - Resnet-v1 and v2 Architectures in Keras using the Functional API. 
The paper on these architectures is available at <a href="https://arxiv.org/pdf/1602.07261v1.pdf">"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning".</a>

The models are plotted and shown in the architecture sub folder. 
Due to lack of suitable training data (ILSVR 2015 dataset) and limited GPU processing power, the weights are not provided.

Contains : Inception v4, Inception-ResNet-v1 and Inception-ResNet-v2

Available at : <a href="https://github.com/titu1994/Inception-v4">Inception v4 in Keras </a>

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

# <a href="https://github.com/titu1994/DenseNet">DenseNet in Keras</a>
Implementation of DenseNet from the paper [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993v1.pdf).

## Usage

1. Run the cifar10.py script to train the DenseNet 40 model
2. Comment out the model.fit_generator(...) line and uncomment the model.load_weights("weights/DenseNet-40-12-CIFAR10.h5") line to test the classification accuracy.

Available at : <a href="https://github.com/titu1994/DenseNet">DenseNet in Keras</a>

# <a href='https://github.com/titu1994/Snapshot-Ensembles'>Snapshot Ensembles in Keras</a>
Implementation of the paper [Snapshot Ensembles](https://github.com/titu1994/Snapshot-Ensembles)

## Usage
The technique is simple to implement in Keras, using a custom callback. These callbacks can be built using the SnapshotCallbackBuilder class in snapshot.py. Other models can simply use this callback builder to other models to train them in a similar manner.

1. Download the 6 WRN-16-4 weights that are provided in the Release tab of the project and place them in the weights directory
2. Run the train_cifar_10.py script to train the WRN-16-4 model on CIFAR-10 dataset (not required since weights are provided)
3. Run the predict_cifar_10.py script to make an ensemble prediction.

Available at : <a href='https://github.com/titu1994/Snapshot-Ensembles'>Snapshot Ensembles in Keras</a>
