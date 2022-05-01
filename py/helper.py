from py import datasets
from sklearn.metrics import ConfusionMatrixDisplay
from enum import Enum
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

def printAccuracy(model,x_test,y_test):
    score = model.evaluate(x_test, y_test, batch_size=100)
    print("Overall Accuracy:", score[1]*100,'%')

def plot_confusion_matrix(model,x_test,y_test):
    pred = model.predict(x_test, batch_size=100)
    confusion_matrix = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis=1),np.argmax(pred, axis=1))
    confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix, display_labels=range(10))
    fig, ax = plt.subplots(figsize=(40,40))
    confusionMatrixDisplay.plot(ax=ax,cmap = plt.get_cmap('Blues'), xticks_rotation='vertical')

class Dataset(Enum):
    MNIST = 1
    CIFAR = 2
class Optimizer(Enum):
    Adam = 1
    SGD = 2
class Runtime(Enum):
    local = 1
    colab = 2

def getOptimizer(opt:Optimizer,learning_rate=0.01):
    if opt == Optimizer.SGD: return keras.optimizers.SGD(learning_rate=learning_rate)
    if opt == Optimizer.Adam: return keras.optimizers.Adam(learning_rate=learning_rate)

def getDataset(set:Dataset):
    if set == Dataset.MNIST: return datasets.getMnist()
    if set == Dataset.CIFAR: return datasets.getCifar()

def CheckpointCallback(path):
    return ModelCheckpoint(filepath=path,verbose=1,save_weights_only=True,monitor='val_accuracy')