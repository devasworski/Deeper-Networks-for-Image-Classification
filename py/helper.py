try:
    from py import datasets
except ImportError:
    import datasets
from sklearn.metrics import ConfusionMatrixDisplay
from enum import Enum
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import os.path
from os import path

# > The `Dataset` class is an enumeration of the two datasets we will be using in this assignment
class Dataset(Enum):
    MNIST = 'MNIST'
    CIFAR = 'CIFAR'
# > An enumeration of the optimizers that can be used to train the neural network
class Optimizer(Enum):
    Adam = 'Adam'
    SGD = 'SGD'
# > The `Runtime` class is an enumeration of the two possible runtimes: `local` and `colab`
class Runtime(Enum):
    local = 1
    colab = 2

def printAccuracy(model,x_test,y_test,index):
    """
    It takes in a model, a test set, and a test label set, and prints out the overall accuracy of the
    model
    
    :param model: The model we're using
    :param x_test: the test data
    :param y_test: The actual labels of the test data
    """
    score = model.evaluate(x_test, y_test, batch_size=100)
    print("Overall Accuracy:", score[index]*100,'%')


def plot_confusion_matrix(model,x_test,y_test, set:Dataset, Modelname:str, opt:Optimizer):
    """
    It takes a model, a test set, and a test label set, and plots a confusion matrix
    
    :param model: the model you want to plot the confusion matrix for
    :param x_test: the test data
    :param y_test: the test labels
    """
    title = 'model: '+Modelname+' data: '+str(set.value)+' opt: '+str(opt.value)
    labels = range(10 )if set == Dataset.MNIST else ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    pred = model.predict(x_test, batch_size=100)
    confusion_matrix = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis=1),np.argmax(pred[0], axis=1),normalize='true')
    confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(20,20))
    label_font = {'size':'18'}
    plt.rcParams.update({'font.size': 14})
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('Observed labels', fontdict=label_font)
    ax.set_title('Confusion Matrix '+title, fontdict={'size':'22'})
    ax.tick_params(axis='both', which='major', labelsize=14)
    confusionMatrixDisplay.plot(ax=ax,cmap = plt.get_cmap('Blues'), xticks_rotation='vertical')


def getOptimizer(opt:Optimizer,learning_rate=0.01):
    """
    > If the optimizer is SGD, return the SGD optimizer with the given learning rate. If the optimizer
    is Adam, return the Adam optimizer with the given learning rate
    
    :param opt: The optimizer to use
    :type opt: Selected Optimizer
    :param learning_rate: The learning rate for the optimizer
    :return: The optimizer object
    """
    if opt == Optimizer.SGD: return keras.optimizers.SGD(learning_rate=learning_rate)
    if opt == Optimizer.Adam: return keras.optimizers.Adam(learning_rate=learning_rate)

def getDataset(set:Dataset):
    """
    > It returns the MNIST dataset if the argument is `Dataset.MNIST`, and the CIFAR dataset if the
    argument is `Dataset.CIFAR`
    
    :param set: The dataset to use
    :type set: Selected Dataset
    :return: A tuple of the training and testing data
    """
    if set == Dataset.MNIST: return datasets.getMnist()
    if set == Dataset.CIFAR: return datasets.getCifar()

def mkdir_if_not_exist(dir):
    if not path.exists(dir):
        os.mkdir(dir[:-20])

def CheckpointCallback(path):
    """
    It creates a callback that saves the model's weights
    
    :param path: The path to save the model file
    :return: A function that takes in a path and returns a ModelCheckpoint object.
    """
    mkdir_if_not_exist(path)
    return ModelCheckpoint(filepath=path,verbose=1,save_weights_only=True)