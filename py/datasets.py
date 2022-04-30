from keras.datasets import mnist
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils

num_classes=10
final_shape=(220,220)
original_shape = (28,28)

def reshape_dims_gray(data):
        (x,y) = data
        x = x.astype('float32')
        x = x.reshape(x.shape[0], original_shape[0], original_shape[1], 1)
        pad_0 = int((final_shape[0]-original_shape[0])/2)
        pad_1 = int((final_shape[1]-original_shape[1])/2)
        x = np.pad(x, ((0,0),(pad_0,pad_1),(pad_0,pad_1),(0,0)), 'constant')
        x = np.stack((x,)*3, axis=-1)
        x = x[:,:,:,0,:]      
        y = np_utils.to_categorical(y, num_classes)
        return x, y

def reshape_dims_rgb(data):
        (x,y) = data
        x = x.astype('float32')
        pad_0 = int((final_shape[0]-original_shape[0])/2)
        pad_1 = int((final_shape[1]-original_shape[1])/2)
        x = np.pad(x, ((0,0),(pad_0,pad_1),(pad_0,pad_1),(0,0)), 'constant')
        y = np_utils.to_categorical(y, num_classes)
        return x, y

def getCifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, shuffle=True ,random_state=42)
    train = (x_train,y_train)
    test = (x_test,y_test)
    val = (x_val,y_val)
    return map(reshape_dims_rgb, [train, test, val])


def getMnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, shuffle=True ,random_state=42)
    train = (x_train,y_train)
    test = (x_test,y_test)
    val = (x_val,y_val)
    return map(reshape_dims_gray, [train, test, val])