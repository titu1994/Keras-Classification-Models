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
