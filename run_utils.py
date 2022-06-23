import torch
import torch.nn as nn
import numpy as np
import time
import torch.utils.data as data
import torch.optim as optim
from torchvision import models

from train import Trainer
import ood_utils
import layer_utils
import pdb


def weights_init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

def run(trainer,args, trainloader,valloader, testloader, classes_idx_OOD, classes,classes_idx_ID, idx):

    if args.load_checkpoint:
        if len(classes_idx_OOD) == 1:
            checkpoint = torch.load(args.save_path+'/model_{}.pt'.format(classes[classes_idx_OOD[0]]))
        else:
            checkpoint = torch.load(args.save_path+'/model_{}.pt'.format(idx))
        trainer.model.load_state_dict(checkpoint)

    else:
        trainer.model.apply(weights_init_)


    if args.train:
        prev_loss = 1e30
        prev_loss_g = 1e30
        for z in range(0,1):
            print("TRAINING")
            if args.dataset1 == 'cifar10_old':
                alexnet = models.vgg16(pretrained=True)
                output_dim = args.ID_tasks
                alexnet.classifier[3] = nn.Linear(4096,1024)
                alexnet.classifier[6] = nn.Linear(1024, output_dim)
                alexnet_dict = alexnet.state_dict()
                model_dict = trainer.model.state_dict()
                pretrained_dict = {k : v for k,v in alexnet_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                trainer.model.load_state_dict(model_dict)
                #torch.nn.init.xavier_normal_(trainer.model.classifier[6].weight, gain =1)
                #torch.nn.init.xavier_normal_(trainer.model.classifier[3].weight, gain = 1)
                scheduler = optim.lr_scheduler.StepLR(trainer.optimizer, step_size = 4, gamma = 0.5)

            for epoch in range(args.epochs):
                start_time = time.time()
                train_loss, train_acc = trainer.optimize(trainloader,classes_idx_OOD, classes_idx_ID)
                end_time = time.time()
                print(f'\t EPOCH: {epoch+1:.0f} | time elapsed: {end_time - start_time:.3f}')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
                loss, acc, act_avg_test,_,_ = trainer.evaluate(valloader,classes_idx_OOD,classes_idx_ID, extract_act = False)
                print(f'\tTest Loss: {loss:.3f} | Test Acc: {acc*100:.2f}%')
                if args.dataset1 =='cifar10':
                    scheduler.step()
                if loss < prev_loss:
                    prev_loss = loss
                    train_loss, train_acc = trainer.optimize(valloader,classes_idx_OOD, classes_idx_ID)
                    if len(classes_idx_OOD) == 1:
                        torch.save(trainer.model.state_dict(), arg.save_path+'/model_{}.pt'.format(classes[classes_idx_OOD[0]]))
                    else:
                        torch.save(trainer.model.state_dict(), args.save_path+'/model_{}.pt'.format(idx))

    else:
            ## do testing
        print("TESTING")


def do_ood_eval(trainloader,valloader, testloader,testset, ood_trainset_list,ood_testset_list, trainer, classes, classes_idx_OOD,args,classes_idx_ID, save_k = 0):

    ## the current implementation only works with a batch size of 1 which is probably not ideal

    ## 1. use traindata to compute thresholds for each class

    _, _, _,activations_list_train, _ = trainer.evaluate(trainloader, classes_idx_OOD, classes_idx_ID, extract_act = True)

    if len(classes_idx_OOD) == 1:
        np.savez(arg.save_path+'/activations/act_full_train_{}.npz'.format(classes[classes_idx_OOD[0]]),**activations_list_train)
    else:
        np.savez(args.save_path+'/activations/act_full_train_{}.npz'.format(save_k),**activations_list_train)

    if len(classes_idx_OOD) == 1:
        activations_list_train = dict(np.load(args.save_path+'/activations/act_full_train_{}.npz'.format(classes[classes_idx_OOD[0]]), allow_pickle = True))
    else:
        activations_list_train = dict(np.load(args.save_path+'/activations/act_full_train_{}.npz'.format(save_k),allow_pickle = True))

   ## compute class wise thresholds
    thresholds = ood_utils.compute_per_class_thresholds(activations_list_train, trainer, classes,classes_idx_OOD, args.ID_tasks, args.baseline_ood)


    if not args.cont_learner:
            ## 2. with these thresholds evuate the test set accuracy
        test_acc = ood_utils.compute_test_Acc(testloader, thresholds, trainer, classes, classes_idx_OOD, args.save_path, args.ID_tasks,classes_idx_ID, args.baseline_ood, save_k)
        # test_loss, test_accuracy = evaluate_with_thresh(testloader, thresholds, trainer.model)
        # ## 3. evaluate ood detection data ## with varying data amounts
        ood_acc = ood_utils.compute_ood_Acc(ood_trainset_list, thresholds, trainer, classes, classes_idx_OOD, args.save_path, args.ID_tasks, args.multiple_dataset,classes_idx_ID,args.baseline_ood,save_k)


        return test_acc, ood_acc

    else:
        avg_act_all_layers, layer_indices = layer_utils.return_all_layer_activations(trainer, testloader)

        test_acc_full = []
        ood_acc_full = []

        lr_mult = 2 # set to 2 for fmnist
        for k in range(0, len(ood_trainset_list),1):


            curr_ood_data = ood_trainset_list[k]


            # loss, acc, activations,activations_list_test, labels_list = trainer.evaluate(testloader, [classes_idx_OOD[k],0],classes_idx_ID, extract_act = False)
            #
            # print(f'\tTest Loss: {loss:.3f} | Test Acc: {acc*100:.2f}%')

            ## incremental test data ood detetion
            if args.full_pipeline:
                testloader = data.DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 2)
                for p in range(0,3):
                    percent = 0.1*p
                    test_ood_acc = ood_utils.compute_incremental_test_acc(testloader, thresholds, trainer, classes, classes_idx_OOD, args.save_path,  len(classes_idx_ID),classes_idx_ID,percent,save_k)
                    print("total in OOD accuracy for sample test data", test_ood_acc)
                sample_data_len = int(0.1*len(curr_ood_data))
                rem_data_len = len(curr_ood_data) - sample_data_len
                sample_ood_data, remaining_ood_data = data.random_split(curr_ood_data,[sample_data_len, rem_data_len])
                ood_acc = ood_utils.compute_ood_Acc([sample_ood_data], thresholds, trainer, classes,[classes_idx_OOD[k]], args.save_path, len(classes_idx_ID), args.multiple_dataset, classes_idx_ID, 0)
                print("ood_acc  :", ood_acc)

            else:
                ood_acc = 100

            if ood_acc > 30:
            # pdb.set_trace()
                batch_size = 32
                classes_idx_ID = np.array(np.insert(classes_idx_ID,len(classes_idx_ID),classes_idx_OOD[k]))
                #
                testset = [testset, ood_testset_list[k]]
                testset = data.ConcatDataset(testset)

                testloader = data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 2)
                current_act_avg_layers, test_acc_list,ood_acc_list,trainer = ood_utils.continual_learner(trainer, curr_ood_data, ood_testset_list[k], testloader, avg_act_all_layers, layer_indices, batch_size,classes_idx_OOD[k],classes_idx_ID,lr_mult)

                test_acc_full.append(test_acc_list)
                ood_acc_full.append(ood_acc_list)

                ood_trainloader = data.DataLoader(curr_ood_data, batch_size = 1, shuffle = False, num_workers = 2)
                _, _, _, activations_list_new_class = trainer.ood_evaluate(ood_trainloader)

                thresholds = ood_utils.update_thresholds(thresholds, activations_list_new_class,trainer,classes_idx_ID)

                new_avg = current_act_avg_layers
                #
                for i in range(len(new_avg)-1):
                   # new_avg[i] =  (current_act_avg_layers[i] + avg_act_all_layers[i]*(1/(lr_mult*10)))
                   new_avg[i] =  (current_act_avg_layers[i] + avg_act_all_layers[i])
                avg_act_all_layers = new_avg
                lr_mult = 2 # 2 for fmnist


        # np.savez('cont_learner_with_another_nosp.npz', test_acc=test_acc_full, ood_acc= ood_acc_full)
        if args.full_pipeline:
            testloader = data.DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 2)
            for p in range(0,3):
                percent = 0.1*p
                test_ood_acc = ood_utils.compute_incremental_test_acc(testloader, thresholds, trainer, classes,classes_idx_OOD, args.save_path, len(classes_idx_ID),classes_idx_ID, percent,save_k)
                print("total in OOD accuracy for sample test data", test_ood_acc)
        print("Exiting Continual Learner experiment")
        exit()  ## the continula learner exits after running 1 experiment
