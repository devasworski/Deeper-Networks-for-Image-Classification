from keras.datasets import mnist
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils
from cv2 import cv2

num_classes=10
final_shape=(224,224)

def scale(data):
        """
        It takes in a batch of images and resizes them to the final shape of the image
        
        :param data: This is the data that we want to scale
        :return: the scaled images and the labels.
        """
        (images,y) = data
        img_rgb = []
        for img in images:
                img = cv2.resize(img, final_shape, interpolation = cv2.INTER_AREA) 
                img_rgb.append(img)
        img_rgb = np.asanyarray(img_rgb)
        y = np_utils.to_categorical(y, num_classes)
        return img_rgb,y

# based on https://fantashit.com/transfer-learning-vgg16-using-mnist/
def to_rgb(data):
        """
        It takes the data, resizes it to the final shape, and converts it to RGB
        
        :param data: the data to be transformed
        :return: the images and the labels.
        """
        (images,y) = data
        img_rgb = []
        for img in images:
                img = cv2.resize(img, final_shape, interpolation = cv2.INTER_AREA)
                #img_rgb.append(np.asarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)))
                img_rgb.append(np.asarray(np.dstack((img, img, img)), dtype=np.uint8))
        img_rgb = np.asanyarray(img_rgb)
        y = np_utils.to_categorical(y, num_classes)
        return img_rgb,y

def getCifar():
        """
        It loads the CIFAR-10 dataset, splits it into training, validation, and test sets, and scales images
        :return: a tuple of tuples. Each tuple contains the data and the labels.
        """
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, shuffle=True ,random_state=42)
        train = (x_train,y_train)
        test = (x_test,y_test)
        val = (x_val,y_val)
        return map(scale, [train, test, val])


def getMnist():
        """
        It loads the MNIST dataset, splits it into training, validation and test sets, and converts the
        images to RGB
        :return: a tuple of tuples. Each tuple contains the data and the labels.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, shuffle=True ,random_state=42)
        train = (x_train,y_train)
        test = (x_test,y_test)
        val = (x_val,y_val)
        return map(to_rgb, [train, test, val])
