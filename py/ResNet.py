# code based on https://medium.com/towards-data-science/building-a-resnet-in-keras-e8f1322a49ba
# and https://machinelearningknowledge.ai/keras-implementation-of-resnet-50-architecture-from-scratch/

from keras.layers import Input, Conv2D, BatchNormalization, Add, AveragePooling2D, Activation, ZeroPadding2D, MaxPooling2D
from keras import Model

def identity_block(X, f, F1, F2, F3):
    """
    **The identity block is the standard block used in ResNets, and corresponds to the case where the
    input activation (say ^{[l]}$) has the same dimension as the output activation (say
    ^{[l+2]}$).**
    
    The details of the architecture are:
    
    - The middle layer of the convolutional block is having a ReLU activation.
    - The first layer of the convolutional block uses the linear activation function.
    - The shortcut should be initialized as a identity matrix
    
    :param X: input tensor
    :param f: filter size
    :param F1: Number of filters in the first 1x1 convolution
    :param F2: number of filters in the second convolutional layer
    :param F3: number of filters in the 3rd layer
    :return: The output of the identity block.
    """
   
    x = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)

    x = Add()([x, X])
    x = Activation('relu')(x)

    return x

def convolutional_block(X, f, F1, F2, F3, s=2):
    """
    The function takes as input an input tensor X and the filter size f. It also takes as input the
    number of filters in the CONV2D layer in the main path, the number of filters in the CONV2D layers
    in the shortcut path, and the stride s
    
    :param X: input tensor
    :param f: filter size
    :param F1: Number of filters in the first 1x1 convolution
    :param F2: number of filters in the second convolutional layer
    :param F3: number of filters in the third convolutional layer
    :param s: stride, defaults to 2 (optional)
    :return: The output of the convolutional block.
    """

    x = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid')(X)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization(axis=3,)(x)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)

    x = Add()([x, X])
    x = Activation('relu')(x)

    return x

def resnet(classes=10, name="ResNet"):
    """
    The function takes in an input image and returns a Keras model with the ResNet
	architecture
    
    :param classes: number of classes in the dataset, defaults to 10 (optional)
    :param name: The name of the model, defaults to ResNet (optional)
    :return: The model is being returned.
    """
    input_layer = Input(shape = (224, 224, 3))

    x = ZeroPadding2D((3, 3))(input_layer)

    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = convolutional_block(x, 3, 64, 64, 256, s=1)
    x = identity_block(x, 3, 64, 64, 256)
    x = identity_block(x, 3, 64, 64, 256)

    x = convolutional_block(x, 3, 128, 128, 512)
    x = identity_block(x, 3, 128, 128, 512)
    x = identity_block(x, 3, 128, 128, 512)
    x = identity_block(x, 3, 128, 128, 512)

    x = convolutional_block(x, 3, 256, 256, 1024)
    x = identity_block(x, 3, 256, 256, 1024)
    x = identity_block(x, 3, 256, 256, 1024)
    x = identity_block(x, 3, 256, 256, 1024)
    x = identity_block(x, 3, 256, 256, 1024)
    x = identity_block(x, 3, 256, 256, 1024)

    x = x = convolutional_block(x, 3, 512, 512, 2048)
    x = identity_block(x, 3, 512, 512, 2048)
    x = identity_block(x, 3, 512, 512, 2048)

    x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)

    return Model(input_layer, x, name=name)