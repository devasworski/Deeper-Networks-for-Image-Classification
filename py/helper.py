from py import datasets
from sklearn.metrics import ConfusionMatrixDisplay
from enum import Enum
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

def printAccuracy(model,x_test,y_test):
    """
    It takes in a model, a test set, and a test label set, and prints out the overall accuracy of the
    model
    
    :param model: The model we're using
    :param x_test: the test data
    :param y_test: The actual labels of the test data
    """
    score = model.evaluate(x_test, y_test, batch_size=100)
    print("Overall Accuracy:", score[4]*100,'%')


def plot_confusion_matrix(model,x_test,y_test):
    """
    It takes a model, a test set, and a test label set, and plots a confusion matrix
    
    :param model: the model you want to plot the confusion matrix for
    :param x_test: the test data
    :param y_test: the test labels
    """
    pred = model.predict(x_test, batch_size=100)
    confusion_matrix = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis=1),np.argmax(pred[0], axis=1))
    confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix, display_labels=range(10))
    fig, ax = plt.subplots(figsize=(40,40))
    confusionMatrixDisplay.plot(ax=ax,cmap = plt.get_cmap('Blues'), xticks_rotation='vertical')

# > The `Dataset` class is an enumeration of the two datasets we will be using in this assignment
class Dataset(Enum):
    MNIST = 1
    CIFAR = 2
# > An enumeration of the optimizers that can be used to train the neural network
class Optimizer(Enum):
    Adam = 1
    SGD = 2
# > The `Runtime` class is an enumeration of the two possible runtimes: `local` and `colab`
class Runtime(Enum):
    local = 1
    colab = 2


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

def CheckpointCallback(path):
    """
    It creates a callback that saves the model's weights
    
    :param path: The path to save the model file
    :return: A function that takes in a path and returns a ModelCheckpoint object.
    """
    return ModelCheckpoint(filepath=path,verbose=1,save_weights_only=True)