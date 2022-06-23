import torch
import torch.nn as nn
import numpy as np
import time
import pdb
import data_loader
from train import Trainer
import random

from arguments import get_args, print_args
import run_utils, data_utils
import json
import argparse
import os

load_args = False


if load_args:
    print("Loading arguments")
    parser =argparse.ArgumentParser()
    args = parser.parse_args()
    args = get_args()
    with open(args.save_path+'/args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    args.batch_size = 1
    args.train = False
    args.load_checkpoint = True
    print_args(args)
     ## this is to make sure you set the arguement for OOD detection

else:
    args = get_args()
    print_args(args)
    if args.train:
        with open(args.save_path+'/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

if not os.path.exists(args.save_path+'/activations'):
    os.makedirs(args.save_path+'/activations')

data_path = '../data' ## this might vary for you

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

#torch.manual_seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)
torch.backends.cudnn.deterministic = True


if args.experiment == "ood_detect":
    print(args.multiple_dataset)
    if not args.multiple_dataset:

        if args.single_ood_class:
            ## do round robin ood ood_detect
            output_dim = args.total_tasks - 1
            test_acc_list = []
            ood_acc_list = []
            for ood_class_idx in range(0, args.total_tasks):

                trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_utils.single_dataset_loader(args.dataset1, data_path, args.batch_size, [ood_class_idx])
                trainer = Trainer(output_dim, device, args)
                print("OOD Class :", classes[ood_class_idx])
                run_utils.run(trainer,args,trainloader, valloader, testloader,[ood_class_idx], classes,[],0)
                # TODO ood detection - from distribution_evaluation - DONE
                if args.load_checkpoint:
                    test_acc, ood_acc = run_utils.do_ood_eval(trainloader, testloader,testset, ood_trainset_list,ood_testset_list, trainer, classes, [ood_class_idx], args,[])
                    test_acc_list.append(test_acc)
                    ood_acc_list.append(ood_acc)
                print("Mean TEST acc :", np.mean(test_acc_list))
                print("Mean OOD acc:", np.mean(ood_acc_list))
        else:

            print("More than one OOD class in the same dataset, this is set-up for continual learning")
            #ood_class_idx = np.arange(args.ID_tasks, args.total_tasks)
            output_dim = args.ID_tasks
            if args.load_list:
                # class_list = []
                # list_idx = np.arange(0,args.total_tasks)
                # class_list.append(list_idx)
                class_list = np.load('class_list1.npz', allow_pickle = True)['class_list']
            else:
                list_idx = np.arange(0,args.total_tasks)
                class_list = []
            ood_acc_list = []
            test_acc_list = []
            for exp_no in range(0,10):
                print("EXP :", exp_no)
                if args.load_list:
                    list_idx = class_list[exp_no]
                else:
                    np.random.shuffle(list_idx)
                    class_list.append(list_idx.copy())
                classes_idx_OOD = list_idx[args.ID_tasks : args.total_tasks]
                classes_idx_ID = list_idx[0:args.ID_tasks]
                trainloader,valloader, testloader, ood_trainset_list, ood_testset_list, classes, testset = data_utils.single_dataset_loader(args.dataset1, data_path, args.batch_size, classes_idx_OOD)
                trainer = Trainer(output_dim, device, args)
                print(classes)
                print("OOD Class :", np.array(classes)[classes_idx_OOD])
                print("IND Class :", np.array(classes)[classes_idx_ID])
                run_utils.run(trainer, args, trainloader, valloader, testloader,classes_idx_OOD, classes,classes_idx_ID,exp_no)
                if args.load_checkpoint:
                    test_acc, ood_acc = run_utils.do_ood_eval(trainloader, valloader,testloader, testset, ood_trainset_list,ood_testset_list, trainer, classes,classes_idx_OOD,args,classes_idx_ID,exp_no)
                    print("TEST_Acc:", test_acc)
                    print("OOD_acc:", ood_acc)
                    ood_acc_list.append(ood_acc)
                    test_acc_list.append(test_acc)
                    if exp_no == 9:
                        combined_acc_list = np.add(ood_acc_list, test_acc_list)/2
                        print("MEAN OOD ACC :",np.mean(ood_acc_list))
                        print("STD OOD ACCC:", np.std(ood_acc_list))
                        print("MEAN TEST ACC :", np.mean(test_acc_list))
                        print("STD TEST ACC:", np.std(test_acc_list))
                        print("MEAN COMBINED ACC :", np.mean(combined_acc_list))
                        print("STD COMBINED ACC :", np.std(combined_acc_list))

            if not args.load_list:
                np.savez('class_list_GTSRB.npz', class_list = class_list)

    else:
        print("OOD detection across datasets")
        output_dim = args.total_tasks ## this if for dataet1 -ID dataset
        trainloader,valloader, testloader,classes,testset, testset_ood,classes_ood = data_utils.mutliple_dataset_loader( data_path, args.dataset1, args.dataset2, args.batch_size)
        print("In dist classes", classes)
        print("OOD dist classes", classes_ood)
        classes_idx_OOD = np.arange(0,len(classes_ood))
        classes_idx_ID = np.arange(0,len(classes))
        trainer = Trainer(output_dim,device, args)
        run_utils.run(trainer, args, trainloader, valloader, testloader,classes_idx_OOD, classes,[],0)
        if args.load_checkpoint:
            test_acc, ood_acc = run_utils.do_ood_eval(trainloader,valloader, testloader,testset, [testset_ood],[testset_ood], trainer, classes, classes_idx_OOD, args, classes_idx_ID,0)
            print("combined acc:", (test_acc+ood_acc)/2)
