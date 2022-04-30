
from keras.datasets import mnist
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split

def getCifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, shuffle=True ,random_state=42)
    #TODO change dimentions of dataset to fit model
    return x_train, y_train, x_val, y_val, x_test, y_test


def getMnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, shuffle=True ,random_state=42)
    #TODO change dimentions of dataset to fit model
    return x_train, y_train, x_val, y_val, x_test, y_test