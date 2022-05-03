# parts of the code in this notebook have been copied from here: https://github.com/rasbt/stat453-deep-learning-ss21/tree/main/L14
import torch
import time
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from itertools import product
try:
    from py import helper
    from py import datasets_torch
except ImportError:
    import helper
    import datasets_torch

def compute_accuracy(model, data_loader, device):
    """
    > The function takes in a model, a data loader, and a device, and returns the accuracy of the model
    on the data loader. It has been taken of the internet.
    
    :param model: the model we are training
    :param data_loader: the data loader for the dataset to test against
    :param device: This is the device on which the model will be trained
    :return: The percentage of correct predictions
    """
    with torch.no_grad():
        correct_predictions, counter = 0, 0
        for _, (img, label) in enumerate(data_loader):
            img = img.to(device)
            label = label.float().to(device)
            logits = model(img)
            _, predicted_label = torch.max(logits, 1)
            counter += label.size(0)
            correct_predictions += (predicted_label == label).sum()
    return correct_predictions.float()/counter * 100

def compute_confusion_matrix(model, data_loader, device):
    """
    It takes a model, a data loader, and a device, and returns a confusion matrix. It has been taken of the internet.
    
    :param model: the model you want to evaluate
    :param data_loader: the data loader for the dataset to test
    :param device: the device to run the model on
    :return: The confusion matrix
    """
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for _, (img, label) in enumerate(data_loader):
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            _, predicted_label = torch.max(logits, 1)
            all_labels.extend(label.to('cpu'))
            all_predictions.extend(predicted_label.to('cpu'))
    all_predictions = all_predictions
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)        
    class_labels = np.unique(np.concatenate((all_labels, all_predictions)))
    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([0, class_labels[0]])
        else:
            class_labels = np.array([class_labels[0], 1])
    n_labels = class_labels.shape[0]
    lst = []
    z = list(zip(all_labels, all_predictions))
    for combi in product(class_labels, repeat=2):
        lst.append(z.count(combi))
    mat = np.asarray(lst)[:, None].reshape(n_labels, n_labels)
    return mat

def getTorchDevice():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def printAccuracy(model,test_loader):
    """
    > It takes a model and a test data loader and prints the overall accuracy of the model on the test
    data
    
    :param model: The model to be tested
    :param test_loader: the test data loader
    """
    test_acc = compute_accuracy(model, test_loader, device=getTorchDevice())
    print(f'Overall Accuracy: {test_acc.to(getTorchDevice()) :.2f}%')

def plot_confusion_matrix_torch(conf_mat, labels, title):
    """
    It takes a confusion matrix, a list of labels, and a title, and returns a plot of the confusion
    matrix. It has been taken of the internet.
    
    :param conf_mat: the confusion matrix
    :param labels: the labels for the confusion matrix
    :param title: The title of the plot
    :return: A figure and an axis
    """
    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float') / total_samples
    fig, ax = plt.subplots(figsize=(20,20))
    ax.grid(False)
    cmap = plt.cm.Blues
    matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    fig.colorbar(matshow)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
                cell_text = ""
                cell_text += format(conf_mat[i, j], 'd')
                cell_text += "\n" + '('
                cell_text += format(normed_conf_mat[i, j], '.2f') + ')'
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)      
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    label_font = {'size':'18'}
    plt.rcParams.update({'font.size': 14})
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('Observed labels', fontdict=label_font)
    ax.set_title('Confusion Matrix '+title, fontdict={'size':'22'})
    return fig, ax

def plot_confusion_matrix(model,test_loader, set:helper.Dataset, Modelname:str, opt:helper.Optimizer):
    """
    It takes a model, a test set, and a test label set, and plots a confusion matrix
    
    :param model: the model you want to plot the confusion matrix for
    :param x_test: the test data
    :param y_test: the test labels
    """
    title = 'model: '+Modelname+' data: '+str(set.value)+' opt: '+str(opt.value)
    labels = range(10 )if set == helper.Dataset.MNIST else ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    mat = compute_confusion_matrix(model=model, data_loader=test_loader, device=torch.device(getTorchDevice()))
    plot_confusion_matrix_torch(mat, labels=labels, title=title)
    plt.show()


def getOptimizer(opt:helper.Optimizer,model,learning_rate=0.01):
    """
    > If the optimizer is SGD, return the SGD optimizer with the given learning rate. If the optimizer
    is Adam, return the Adam optimizer with the given learning rate
    
    :param opt: The optimizer to use
    :type opt: Selected Optimizer
    :param learning_rate: The learning rate for the optimizer
    :return: The optimizer object
    """
    if opt == helper.Optimizer.SGD: return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if opt == helper.Optimizer.Adam: return torch.optim.Adam(model.parameters(), lr=learning_rate, momentum=0.9)

def train(model, num_epochs, train_loader,valid_loader, test_loader, optimizer):
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(getTorchDevice())
            targets = targets.to(getTorchDevice())
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_acc = compute_accuracy(model, train_loader, device=getTorchDevice())
            valid_acc = compute_accuracy(model, valid_loader, device=getTorchDevice())
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f}% '
                  f'| Validation: {valid_acc :.2f}%')
        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')        
    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')
    test_acc = compute_accuracy(model, test_loader, device=getTorchDevice())
    print(f'Test accuracy {test_acc :.2f}%')

def getDataset(set:helper.Dataset):
    """
    > It returns the MNIST dataset if the argument is `Dataset.MNIST`, and the CIFAR dataset if the
    argument is `Dataset.CIFAR`
    
    :param set: The dataset to use
    :type set: Selected Dataset
    :return: A tuple of the training and testing data
    """
    if set == helper.Dataset.MNIST: return datasets_torch.getMnist()
    if set == helper.Dataset.CIFAR: return datasets_torch.getCifar()