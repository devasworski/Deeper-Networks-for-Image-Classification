from keras.datasets import mnist
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import numpy as np
from cv2 import cv2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision

# https://stackoverflow.com/questions/69967363/scikit-learn-train-test-split-into-pytorch-dataloader

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class TorchDataset(Dataset):

        def __init__(self, data):
                img, labels = data
                img_processed = []
                for i in img:
                        img_processed.append(preprocess(i))
                self.x_data = img_processed
                self.y_data = labels
        def __getitem__(self, i):
                return self.x_data[i], self.y_data[i]
        def __len__(self):
                return len(self.x_data)

num_classes=10
final_shape=(64,64)

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
        return img_rgb,y

def getCifar(batch_size):
        """
        It loads the CIFAR-10 dataset, splits it into training, validation, and test sets, scales the data,
        and returns the data as PyTorch DataLoaders
        
        :param batch_size: The number of images to be passed through the network at once
        :return: train_loader, test_loader, val_loader
        """
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, shuffle=True ,random_state=42)
        train = (x_train,y_train[:,0])
        test = (x_test,y_test[:,0])
        val = (x_val,y_val[:,0])
        train, test, val = map(scale, [train, test, val])
        train_loader = DataLoader(TorchDataset(train),batch_size=batch_size)
        test_loader = DataLoader(TorchDataset(test),batch_size=batch_size)
        val_loader = DataLoader(TorchDataset(val),batch_size=batch_size)
        return train_loader, test_loader, val_loader


def getMnist(batch_size):
        """
        It loads the MNIST dataset, splits it into training, validation and test sets, converts the images
        to RGB, and returns a data loader for each of the three sets
        
        :param batch_size: The number of images to be passed through the network at once
        :return: train_loader, test_loader, val_loader
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, shuffle=True ,random_state=42)
        train = (x_train,y_train)
        test = (x_test,y_test)
        val = (x_val,y_val)
        train, test, val = map(to_rgb, [train, test, val])
        train_loader = DataLoader(TorchDataset(train),batch_size=batch_size)
        test_loader = DataLoader(TorchDataset(test),batch_size=batch_size)
        val_loader = DataLoader(TorchDataset(val),batch_size=batch_size)
        return train_loader, test_loader, val_loader

