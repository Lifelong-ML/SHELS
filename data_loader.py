import numpy as np

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import pdb
from custom_dataloader import CustomDataset

import scipy


def data_loader_CIFAR_10(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224])]) # was 224, 224
                                    #transforms.Normalize( (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    # transform= transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = datasets.CIFAR10(root = '../data', train = True, download = True, transform = transform)
    testset  = datasets.CIFAR10(root = '../data', train = False, download = True, transform = transform)
    classes = trainset.classes
    print(classes)
    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0
    if len(ood_class) > 0:
        trainset, testset = create_traintestset(ood_class, trainset,testset)

        for i in range(0, len(ood_class)):
            trainset_ood = datasets.CIFAR10(root = '../data', train = True, download = True, transform = transform)
            testset_ood  = datasets.CIFAR10(root = '../data', train = False, download = True, transform = transform)
            ood_trainset, ood_testset = create_OOD_dataset(ood_class[i], trainset_ood,testset_ood)
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)

    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])
    # combine_set = []
    # combine_set.append(val_set)
    # combine_set.append(ood_trainset)
    # combine_set = data.ConcatDataset(combine_set)
    train_set = trainset
    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    # ood_trainloader = data.DataLoader(ood_trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    # oodloader   = data.DataLoader(oodset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)
    #return trainloader, valloader, testloader, ood_trainloader, oodloader, classes
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset


def data_loader_GTSRB(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([112,112])]) # 112,112
                                    #transforms.Normalize( (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    # transform= transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    root_dir_train = '../data/GTSRB/Final_Training/Images/'
    root_dir_test = '../data/GTSRB/Test/'
    trainset = CustomDataset(data_path = root_dir_train,transform = transform)
    testset  = CustomDataset(data_path = root_dir_test,transform = transform)
    classes = trainset.classes
    print(len(classes))
    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0
    if len(ood_class) > 0:
        trainset, testset = create_traintestset(ood_class, trainset,testset)

        for i in range(0, len(ood_class)):
            trainset_ood = CustomDataset(data_path = root_dir_train, transform = transform)
            testset_ood  = CustomDataset(data_path = root_dir_test, transform = transform)
            ood_trainset, ood_testset = create_OOD_dataset(ood_class[i], trainset_ood,testset_ood)
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)

    train_data_len = 2*( int(0.88*len(trainset)/2.0))
    val_data_len = len(trainset) - train_data_len
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])

    # bypassing val dataset


    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
    # ood_trainloader = data.DataLoader(ood_trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    # oodloader   = data.DataLoader(oodset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)
    #return trainloader, valloader, testloader, ood_trainloader, oodloader, classes
    return trainloader,trainloader, testloader, ood_trainset_list, ood_testset_list, classes, testset

def data_loader_SVHN(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):

    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([32,32])]) # was 32 32


    trainset = datasets.SVHN(root = '../data', split = 'train', download = True, transform = transform)
    testset  = datasets.SVHN(root = '../data', split = 'test', download = True, transform = transform)

    classes = [0,1,2,3,4,5,6,7,8,9]
    print(classes)
    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0
    if len(ood_class) > 0:
        trainset, testset = create_traintestset_svhn(ood_class, trainset,testset)

        for i in range(0, len(ood_class)):
            trainset_ood = datasets.SVHN(root = '../data', split = 'train', download = True, transform = transform)
            testset_ood  = datasets.SVHN(root = '../data', split = 'test', download = True, transform = transform)
            ood_trainset, ood_testset = create_OOD_dataset_svhn(ood_class[i], trainset_ood,testset_ood)
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)

    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])
    # combine_set = []
    # combine_set.append(val_set)
    # combine_set.append(ood_trainset)
    # combine_set = data.ConcatDataset(combine_set)

    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    # ood_trainloader = data.DataLoader(ood_trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    # oodloader   = data.DataLoader(oodset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)
    #return trainloader, valloader, testloader, ood_trainloader, oodloader, classes
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset


def data_loader_FashionMNIST(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):

    transform = transforms.Compose([transforms.ToTensor()])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    trainset = datasets.FashionMNIST(root = '../data', train = True, download = True, transform = transform)
    testset  = datasets.FashionMNIST(root = '../data', train = False, download = True, transform = transform)
    classes = trainset.classes
    print(classes)
    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0
    if len(ood_class) > 0:
        trainset, testset = create_traintestset(ood_class, trainset,testset)

        for i in range(0, len(ood_class)):
            trainset_ood = datasets.FashionMNIST(root = '../data', train = True, download = True, transform = transform)
            testset_ood  = datasets.FashionMNIST(root = '../data', train = False, download = True, transform = transform)
            ood_trainset, ood_testset = create_OOD_dataset(ood_class[i], trainset_ood,testset_ood)
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)

    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])
    # combine_set = []
    # combine_set.append(val_set)
    # combine_set.append(ood_trainset)
    # combine_set = data.ConcatDataset(combine_set)

    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    # ood_trainloader = data.DataLoader(ood_trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    # oodloader   = data.DataLoader(oodset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)
    #return trainloader, valloader, testloader, ood_trainloader, oodloader, classes
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset

def data_loader_MNIST(root_dir = './data', BATCH_SIZE = 1, ood_class = [7]):

    transform = transforms.Compose([transforms.ToTensor()])
                                    # normalization values = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    trainset = datasets.MNIST(root = '../data', train = True, download = True, transform = transform)
    testset  = datasets.MNIST(root = '../data', train = False, download = True, transform = transform)
    classes = trainset.classes
    print(classes)
    ood_testset_list = []
    ood_trainset_list = []
    ood_testset_len = 0
    if len(ood_class) > 0:
        trainset, testset = create_traintestset(ood_class, trainset,testset)

        for i in range(0, len(ood_class)):
            trainset_ood = datasets.MNIST(root = '../data', train = True, download = True, transform = transform)
            testset_ood  = datasets.MNIST(root = '../data', train = False, download = True, transform = transform)
            ood_trainset, ood_testset = create_OOD_dataset(ood_class[i], trainset_ood,testset_ood)
            ood_trainset_list.append(ood_trainset)
            ood_testset_list.append(ood_testset)
            ood_testset_len+=len(ood_testset)

    train_data_len = int(0.88*len(trainset))
    val_data_len = len(trainset) - train_data_len
    train_set, val_set = data.random_split(trainset,[train_data_len, val_data_len])
    # combine_set = []
    # combine_set.append(val_set)
    # combine_set.append(ood_trainset)
    # combine_set = data.ConcatDataset(combine_set)

    trainloader = data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    testloader  = data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    valloader   = data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)
    # ood_trainloader = data.DataLoader(ood_trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    # oodloader   = data.DataLoader(oodset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

    print("Train set length", len(train_set))
    print("Val set length", len(val_set))
    print("Test set length", len(testset))
    print("OOD set length", ood_testset_len)
    #return trainloader, valloader, testloader, ood_trainloader, oodloader, classes
    return trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset

def create_subdataset(classes, trainset, testset):

    #for i in range(len(ood_class)):

    idx = np.where(np.isin(np.array(trainset.targets), classes, invert = False))[0].tolist()
    accessed_targets = map(trainset.targets.__getitem__, idx)
    trainset.targets = list(accessed_targets)
    accessed_data = map(trainset.data.__getitem__, idx)
    trainset.data = list(accessed_data)

    idx = np.where(np.isin(np.array(testset.targets), classes, invert = False))[0].tolist()
    accessed_targets = map(testset.targets.__getitem__, idx)
    testset.targets = list(accessed_targets)
    accessed_data = map(testset.data.__getitem__, idx)
    testset.data = list(accessed_data)
    return trainset, testset



def create_traintestset(ood_class, trainset, testset):

    #for i in range(len(ood_class)):

    idx = np.where(np.isin(np.array(trainset.targets), ood_class, invert = True))[0].tolist()
    accessed_targets = map(trainset.targets.__getitem__, idx)
    trainset.targets = list(accessed_targets)
    accessed_data = map(trainset.data.__getitem__, idx)
    trainset.data = list(accessed_data)

    idx = np.where(np.isin(np.array(testset.targets), ood_class, invert = True))[0].tolist()
    accessed_targets = map(testset.targets.__getitem__, idx)
    testset.targets = list(accessed_targets)
    accessed_data = map(testset.data.__getitem__, idx)
    testset.data = list(accessed_data)
    return trainset, testset

def create_OOD_dataset(ood_class, ood_trainset, oodset):


    idx_ood = np.where(np.array(ood_trainset.targets) == ood_class)[0].tolist()
    accessed_targets = map(ood_trainset.targets.__getitem__, idx_ood)
    ood_trainset.targets = list(accessed_targets)
    accessed_data = map(ood_trainset.data.__getitem__, idx_ood)
    ood_trainset.data = list(accessed_data)




    idx_ood = np.where(np.array(oodset.targets) == ood_class)[0].tolist()
    accessed_targets = map(oodset.targets.__getitem__, idx_ood)
    oodset.targets = list(accessed_targets)
    accessed_data = map(oodset.data.__getitem__, idx_ood)
    oodset.data = list(accessed_data)


    return ood_trainset, oodset

def create_traintestset_svhn(ood_class, trainset, testset):

    for i in range(len(ood_class)):


        idx = np.where(np.array(trainset.labels) != ood_class[i])[0].tolist()
        accessed_targets = map(trainset.labels.__getitem__, idx)
        trainset.labels = list(accessed_targets)
        accessed_data = map(trainset.data.__getitem__, idx)
        trainset.data = list(accessed_data)

        idx = np.where(np.array(testset.labels) != ood_class[i])[0].tolist()
        accessed_targets = map(testset.labels.__getitem__, idx)
        testset.labels = list(accessed_targets)
        accessed_data = map(testset.data.__getitem__, idx)
        testset.data = list(accessed_data)

    return trainset, testset

def create_OOD_dataset_svhn(ood_class, ood_trainset, oodset):


    idx_ood = np.where(np.array(ood_trainset.labels) == ood_class)[0].tolist()
    accessed_targets = map(ood_trainset.labels.__getitem__, idx_ood)
    ood_trainset.labels = list(accessed_targets)
    accessed_data = map(ood_trainset.data.__getitem__, idx_ood)
    ood_trainset.data = list(accessed_data)




    idx_ood = np.where(np.array(oodset.labels) == ood_class)[0].tolist()
    accessed_targets = map(oodset.labels.__getitem__, idx_ood)
    oodset.labels = list(accessed_targets)
    accessed_data = map(oodset.data.__getitem__, idx_ood)
    oodset.data = list(accessed_data)


    return ood_trainset, oodset
