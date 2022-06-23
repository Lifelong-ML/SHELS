import numpy as np
from scipy.stats import norm
from tabulate import tabulate

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from weight_freezing import get_weights_mask
import layer_utils
from copy import deepcopy
from scipy.special import softmax

import pdb


def compute_per_class_thresholds(activations_train, trainer, classes, ood_class_idx, in_dist_classes, baseline_ood):

    if len(ood_class_idx) == 1:
        ind_classes =  classes.copy()
        ind_classes.remove(classes[ood_class_idx[0]])
    else:
         ind_classes = classes[0: in_dist_classes]
    weights = trainer.model.classifier[trainer.weight_idx].weight.detach().cpu().numpy()
    thresholds = np.zeros(in_dist_classes)

    total_train_data = 0
    total_true_y = 0
    for class_idx in range(0,in_dist_classes):
        y_list = []
        train_data = activations_train[str(class_idx)]

        total_train_data+=len(train_data)
        true_y = 0
        for idx in range(0,len(train_data)):

            train_feats = train_data[idx,:]
            y_label = []
            y_weighted = []
            for k in range(0, in_dist_classes):


                if baseline_ood:
                    norm = np.linalg.norm(train_feats)
                    norm = np.where(norm==0,1e-20,norm)
                    feat_norm = train_feats/norm

                    weight_norm = weights[k,:]/np.abs(np.linalg.norm(weights[k,:]))
                    weighted_feats = np.sum(feat_norm * weight_norm)
                else:
                    weighted_feats = np.sum(train_feats*weights[k,:])


                y_weighted.append(weighted_feats)

            if np.argmax(y_weighted) == class_idx:
                y_list.append(max(y_weighted))
                true_y +=1

        #print("accuracy {}".format(ind_classes[class_idx]), 100*true_y/len(train_data))

        thresholds[class_idx] = np.mean(y_list) - np.std(y_list)
        total_true_y+=true_y
    total_accuracy =  100*total_true_y/total_train_data
    print("total in distribution accuracy",total_accuracy)
    print("Computed thresholds using training data")
    # print(thresholds)
    return thresholds



def update_thresholds(thresholds,activations_list_new_class,trainer,in_dist_classes):

    weights = trainer.model.classifier[2].weight.detach().cpu().numpy()

    # total_train_data = 0
    # total_true_y = 0
    y_list = []
    true_y = 0
    class_idx = len(in_dist_classes)

    thresholds = np.insert(thresholds,class_idx-1,0)
    for idx in range(0,len(activations_list_new_class)):
        feats = activations_list_new_class[idx]
        # weighted_feats = np.matmul(feats,weights[class_idx-1,:].T)
        # feats = activations_list_new_class[idx]

        weighted_feats = np.matmul(feats, weights.T)
        # print(weighted_feats)
        if np.argmax(weighted_feats) == class_idx-1:
            y_list.append(weighted_feats)
            true_y +=1
        # y_list.append(weighted_feats)
    thresholds[class_idx-1] = np.mean(y_list) - np.std(y_list)

    total_accuracy =  100*true_y/len(activations_list_new_class)
    # print("New class accuracy",total_accuracy)
    print("Updated thresholds using training data")
    # print(thresholds)
    return thresholds


def compute_test_Acc(testloader, thresholds, trainer, classes, ood_class_idx, save_path, in_dist_classes, in_dist_class_list, baseline_ood,save_k = 0):

    _, _, activations,activations_test, _ = trainer.evaluate(testloader, ood_class_idx,in_dist_class_list, extract_act = True)

    if len(ood_class_idx) == 1:
        np.savez(save_path+'/activations/act_full_test_{}.npz'.format(classes[ood_class_idx[0]]),**activations_test)
    else:
        np.savez(save_path+'/activations/act_full_test_{}.npz'.format(save_k),**activations_test)


    if len(ood_class_idx) == 1:
        activations_test = dict(np.load(save_path+'/activations/act_full_test_{}.npz'.format(classes[ood_class_idx[0]]), allow_pickle = True))
    else:
        activations_test = dict(np.load(save_path+'/activations/act_full_test_{}.npz'.format(save_k),allow_pickle = True))

    weights = trainer.model.classifier[trainer.weight_idx].weight.detach().cpu().numpy()
    if len(ood_class_idx) == 1:
        ind_classes =  classes.copy()
        ind_classes.remove(classes[ood_class_idx[0]])
    else:
         ind_classes = classes[0: in_dist_classes]

    total_test_data = 0
    total_true_y = 0
    for class_idx in range(0,in_dist_classes):
        y_list = []
        test_data = activations_test[str(class_idx)]
        total_test_data+=len(test_data)
        true_y = 0
        for idx in range(0,len(test_data)):

            test_feats = test_data[idx,:]

            y_label = []
            y_weighted = []
            for k in range(0, in_dist_classes):

                if baseline_ood:
                    norm = np.linalg.norm(test_feats)
                    norm = np.where(norm==0,1e-20,norm)
                    feat_norm = test_feats/norm
                    weight_norm = weights[k,:]/np.abs(np.linalg.norm(weights[k,:]))
                    weighted_feats = np.sum(feat_norm * weight_norm)
                else:
                    weighted_feats = np.sum(test_feats*weights[k,:])

                y_weighted.append(weighted_feats)
            if max(y_weighted) > thresholds[class_idx] and np.argmax(y_weighted) == class_idx:
                y_list.append(max(y_weighted))
                true_y +=1
        print("accuracy {}".format(ind_classes[class_idx]), 100*true_y/len(test_data))
        total_true_y+=true_y
    total_test_accuracy =  100*total_true_y/total_test_data
    print("total in distribution accuracy", total_test_accuracy)
    return total_test_accuracy

def compute_incremental_test_acc(testloader, thresholds, trainer, classes, ood_class_idx, save_path, in_dist_classes, in_dist_class_list,begin_idx, save_k = 0):

    _, _, activations,activations_test, _ = trainer.evaluate(testloader, ood_class_idx,in_dist_class_list, extract_act = True)

    if len(ood_class_idx) == 1:
        np.savez(save_path+'/activations/act_full_test_{}.npz'.format(classes[ood_class_idx[0]]),**activations_test)
    else:
        np.savez(save_path+'/activations/act_full_test_{}.npz'.format(save_k),**activations_test)


    if len(ood_class_idx) == 1:
        activations_test = dict(np.load(save_path+'/activations/act_full_test_{}.npz'.format(classes[ood_class_idx[0]]), allow_pickle = True))
    else:
        activations_test = dict(np.load(save_path+'/activations/act_full_test_{}.npz'.format(save_k),allow_pickle = True))

    weights = trainer.model.classifier[2].weight.detach().cpu().numpy()

    if len(ood_class_idx) == 1:
        ind_classes =  classes.copy()
        ind_classes.remove(classes[ood_class_idx[0]])
    else:
         ind_classes = classes[0: len(in_dist_class_list)]

    total_test_data = 0
    total_true_y = 0
    for class_idx in range(0,len(in_dist_class_list)):

        test_data = activations_test[str(class_idx)]
        test_data_len = len(test_data)
        first_idx = int(begin_idx*test_data_len)
        last_idx = int((begin_idx+0.1)*test_data_len)
        # sample_len = int(0.1*len(test_data))
        total_test_data+=(last_idx-first_idx)
        true_y = 0
        for idx in range(first_idx,last_idx):

            test_feats = test_data[idx,:]

            weighted_feats = np.matmul(test_feats,weights.T)

            y_weighted = weighted_feats
            index = np.argmax(y_weighted)
            if max(y_weighted) < thresholds[index]: #and index == class_idx:
                true_y +=1
        per_ood_accuracy =  100*true_y/(last_idx-first_idx)
        # print("OOD accuracy {} :".format(ind_classes[class_idx]), per_ood_accuracy)
        total_true_y+=true_y
    total_ood_accuracy =  100*total_true_y/total_test_data
    # print("total in OOD accuracy", total_ood_accuracy)
    return total_ood_accuracy



def compute_ood_Acc(ood_data_list, thresholds, trainer, classes, ood_class_idx, save_path, in_dist_classes, multiple_dataset, in_dist_class_list, baseline_ood ,save_k = 0):
    ood_acc_list = []
    for i in range(0,len(ood_data_list)):

        if True:
            ood_trainloader = data.DataLoader(ood_data_list[i], batch_size = 1, shuffle = False, num_workers = 2)
        else:
            ood_trainloader = ood_data_list[i]
        loss, acc, act_avg_ood, activation_list_ood = trainer.ood_evaluate(ood_trainloader)

        if len(ood_class_idx) > 0:
            np.savez(save_path+'/activations/act_full_ood_{}_{}.npz'.format(classes[ood_class_idx[i]],save_k),act_ood = activation_list_ood)
        else:
            np.savez(save_path+'/activations/act_full_ood_{}.npz'.format(save_k),act_ood = activation_list_ood)


        if len(ood_class_idx) > 0:
            activations_ood = dict(np.load(save_path+'/activations/act_full_ood_{}_{}.npz'.format(classes[ood_class_idx[i]],save_k), allow_pickle = True))['act_ood']
        else:
            activations_ood = dict(np.load(save_path+'/activations/act_full_ood_{}.npz'.format(save_k), allow_pickle = True))['act_ood']

        weights = trainer.model.classifier[trainer.weight_idx].weight.detach().cpu().numpy()
        ood_data = activations_ood
        true_y = 0
        for idx in range(0,len(ood_data)):
            ood_feats = ood_data[idx,:]

            y_weighted = []
            for k in range(0, in_dist_classes):

                if baseline_ood:
                    norm = np.linalg.norm(ood_feats)
                    norm = np.where(norm==0,1e-20,norm)
                    feat_norm = ood_feats/norm

                    #feat_norm = ood_feats/(np.abs(np.linalg.norm(ood_feats)))
                    weight_norm = weights[k,:]/np.abs(np.linalg.norm(weights[k,:]))
                    weighted_feats = np.sum(feat_norm * weight_norm)
                else:
                    weighted_feats = np.sum(ood_feats*weights[k,:])
                y_weighted.append(weighted_feats)
            index = np.argmax(y_weighted)
            if max(y_weighted) <= thresholds[index]:
                true_y +=1
        total_ood_accuracy =  100*true_y/len(ood_data)
        ood_acc_list.append(total_ood_accuracy)
        if multiple_dataset:
            print("OOD accuracy :", total_ood_accuracy)
        else:
            print("OOD accuracy {} :".format(classes[ood_class_idx[i]]), total_ood_accuracy)
    mean_ood_acc = np.mean(ood_acc_list)
    return mean_ood_acc

def continual_learner(trainer, ood_traindata, ood_testdata, testloader, avg_act_all_layers, layer_indices, batch_size, ood_class,  in_dist_classes_list,lr_mult):

    trainer.model.increment_classes(1)
    trainer.optimizer = optim.Adam(trainer.model.parameters(), lr = trainer.learning_rate/lr_mult)

    trainer.output_dim+=1

    ood_trainloader = data.DataLoader(ood_traindata, batch_size = batch_size, shuffle = False, num_workers = 2)
    ood_testloader = data.DataLoader(ood_testdata, batch_size = batch_size, shuffle = False, num_workers = 2)

    mask_c,weights_c,nodes_c, mask_f, weights_f, nodes_f, activations_norm = get_weights_mask(trainer.model,avg_act_all_layers, layer_indices,trainer.output_dim)

    trainer.model.update_model_weights(weights_f, weights_c, nodes_f, nodes_c)
    epochs = 8 #9
    # if trainer.output_dim == 8:
    #     epochs = 0
    old_model = deepcopy(trainer.model) ## has the frozen weights
    # old_weight = trainer.model.classifier[2].weight[2,:].detach()
    prev_loss = 1e30
    test_acc_list = []
    ood_acc_list = []
    loss, acc_test, act_avg_test,_,_ = trainer.evaluate(testloader,[ood_class, ood_class],  in_dist_classes_list, extract_act = False)
    print(f'\tTest Loss: {loss:.3f} | Test Acc: {acc_test*100:.2f}%')
    for epoch in range(epochs):

        train_loss, train_acc = trainer.optimize_cont(ood_trainloader,[ood_class],mask_c,mask_f, nodes_c, nodes_f, weights_c, weights_f,activations_norm, old_model,in_dist_classes_list)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        loss, acc_test, act_avg_test,_,_ = trainer.evaluate(testloader,[ood_class, ood_class],  in_dist_classes_list, extract_act = False)
        print(f'\tTest Loss: {loss:.3f} | Test Acc: {acc_test*100:.2f}%')
        loss_ood, acc_ood, act_avg_test,_,_ = trainer.evaluate(ood_testloader,[ood_class,  ood_class], in_dist_classes_list, extract_act = False)
        print(f'\tOOD Loss: {loss_ood:.3f} | OOD Acc: {acc_ood*100:.2f}%')
        test_acc_list.append(acc_test)
        ood_acc_list.append(acc_ood)


    avg_act_all_layers, layer_indices = layer_utils.return_all_layer_activations(trainer, ood_trainloader)

    return avg_act_all_layers, test_acc_list, ood_acc_list, trainer
