import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from model import Model
from model_cosine_gtsrb import GTSRB
from model_cosine_svhn import SVHN
from model_cosine1 import MNIST
from model_vgg16_cosine import CIFAR10
from model_cosine_cifar import CIFAR
import pdb

class Trainer:
    def __init__(self,output_dim,device, args):

        self.output_dim = output_dim
        self.device = device
        if args.dataset1 == 'mnist':
            self.model = MNIST(self.output_dim, args.cosine_sim, args.baseline)
            no_layers = 8
            ll_layer_idx = 2
        elif args.dataset1 == 'fmnist':
            self.model = MNIST(self.output_dim, args.cosine_sim, args.baseline)
            no_layers = 8
            ll_layer_idx = 2
        elif args.dataset1 == 'svhn':
            self.model = SVHN(self.output_dim, args.cosine_sim, args.baseline)
            no_layers = 9
            ll_layer_idx = 4
        elif args.dataset1 == 'gtsrb':
            self.model = GTSRB(self.output_dim, args.cosine_sim, args.baseline)
            no_layers  = 9
            ll_layer_idx = 4
        elif args.dataset1 == 'cifar10':
            self.model = CIFAR(self.output_dim, args.cosine_sim, args.baseline)
            no_layers =  8 #16
            ll_layer_idx = 2 #6

        self.learning_rate = args.lr
        #self.model_gating = GatingModel()
        self.alpha = 0.001
        self.beta = 0.001
        self.l1_reg = False
        self.sparsity_flag = args.sparsity_es or args.sparsity_gs
        print("sparsity", self.sparsity_flag)
        print("ALPHA :", self.alpha)
        print("BETA :", self.beta)
        self.sparsity_es = args.sparsity_es
        self.sparsity_gs = args.sparsity_gs

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        # if args.dataset1 =='cifar10':
        #     # usef lr = 0.001
        #     self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum = 0.9, weight_decay = 5e-4)
        # else:
        #     self.optimizer = optim.Adam(self.model.parameters(),lr = self.learning_rate)
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr = self.learning_rate, weight_decay = 1e-6)
        #self.optimizer_g = optim.Adam(self.model_gating.parameters(), lr = self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.BCELoss()
        #if not args.baseline:
        #self.criterion = nn.NLLLoss()
        self.model = self.model.to(self.device)
        #self.model_gating = self.model_gating.to(self.device)
        self.criterion = self.criterion.to(self.device)

        ## CGSEG configuration
        self.L = no_layers## number of layers ## for vgg16 -= 16, for conv = 9
        self.weight_idx = ll_layer_idx
        self.penultimate_feats_len = self.model.classifier[self.weight_idx].in_features
        self.mu = 0
        self.delta_mu = 1/(self.L-1)

        ## weight penalties
        self.beta_weight = 3000 #3000 for fmnist
        self.beta_bias = 3000   #3000 for fmnist

    def calculate_accuracy(self,y_pred, y):
        top_pred = y_pred.argmax(1, keepdim = True)

        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc

    def L1_regularisation(self,W):
        return W.norm(1)

    def sparsity(self, W):
        g_l1 = torch.norm(W,p=1,dim=0)  ## l1 norm within each group/node
        g_s = torch.norm(W, p=2,dim=0)
        if self.sparsity_es and not self.sparsity_gs:
            #print("Exclsuive")
            #return self.beta*(self.mu)*torch.norm(W,p=1)
            return self.beta*(self.mu)*(torch.norm(g_l1,p=2,dim = 0)) #+ torch.norm(B, p =1,dim=0))
            #return self.beta*(self.mu/2)*(torch.pow(g_l1,2).sum())
        elif self.sparsity_gs and not self.sparsity_es:
            #print("Group")
            return self.alpha*((1-self.mu)*(torch.norm(g_s,p=1,dim=0)))#+ torch.norm(B, p = 1, dim = 0))
            #return self.alpha*(1-self.mu)*(g_s.sum())
        elif self.sparsity_es and self.sparsity_gs:
            #print("both")
            #g_es = (self.alpha*(1-self.mu)* g_s + self.beta*(self.mu/2)*(g_l1.pow(2))).sum()
            #g_es = (self.alpha*(1-self.mu)*g_s + self.beta*(self.mu/2)*torch.pow(g_l1,2)).sum()
            g_es = self.alpha*((1-self.mu)*(torch.norm(g_s,p=1,dim=0))) + self.beta*(self.mu)*(torch.norm(g_l1,p=2,dim = 0))
            return g_es
        else:
            return 0

    def sparsityconv(self, W):
        g_l1 = torch.norm(W,p=1,dim=(0,2,3))  ## l1 norm within each group/node
        g_s = torch.norm(W, p=2,dim=(0,2,3))
        #g_es = ((1-self.mu)* g_s + (self.mu/2)*(g_l1.pow(2))).sum()
        if self.sparsity_es and not self.sparsity_gs:
            #print("Exclusive Conv")
            #return self.beta*self.mu*torch.norm(W,p=1)
            return  self.beta*(self.mu)*(torch.norm(g_l1,p=2,dim = 0)) #+ torch.norm(B, p =1,dim=0))
            #return self.beta*(self.mu/2)*(torch.pow(g_l1,2).sum())
        elif self.sparsity_gs and not self.sparsity_es:
            #print("group Conv")
            return self.alpha*((1-self.mu)*(torch.norm(g_s,p=1,dim=0))) #+ torch.norm(B, p = 1, dim = 0))
            #return self.alpha*(1-self.mu)*(g_s.sum())
        elif self.sparsity_es and self.sparsity_gs:
            #g_es = (self.alpha*(1-self.mu)* g_s + self.beta*(self.mu/2)*(g_l1.pow(2))).sum()
            #g_es = (self.alpha*(1-self.mu)*g_s + self.beta*(self.mu/2)*torch.pow(g_l1,2)).sum()
            #print("both conv")
            g_es = self.alpha*((1-self.mu)*(torch.norm(g_s,p=1,dim=0))) + self.beta*(self.mu)*(torch.norm(g_l1,p=2,dim = 0))

            return g_es
        else:
            return 0


    def optimize(self,data,ood_class, in_dist_classes):#,mask_c, mask_f, nodes, nodes_f, w, w_f):
        epoch_loss = 0
        epoch_acc = 0

        self.model.train()

        for (x,y) in data:
            x = x.to(self.device)
            y = y.to(self.device)

            self.mu = 0
            self.optimizer.zero_grad()
            layers , y_pred = self.model(x)

            if len(ood_class) == 1:
                y = torch.where(y>ood_class[0], y-1,y)

            elif len(ood_class) > 1:
                y_new = y.clone()
                for j in range(0,len(in_dist_classes)):

                    y_new = torch.where(y == in_dist_classes[j], j, y_new)
                y = y_new.clone()

            loss = self.criterion(y_pred,y)

            acc = self.calculate_accuracy(y_pred,y)
            loss_reg = 0
            loss_theta = 0

            if self.l1_reg:
                for name, W in self.model.named_parameters():
                    if name == 'classifier.2.weight':
                        loss_reg += self.alpha*self.L1_regularisation(W)

            if self.sparsity_flag:

                for i in range(0, len(self.model.features)):
                    if isinstance(self.model.features[i], nn.Conv2d):
                        W = self.model.features[i].weight
                        B = self.model.features[i].bias
                        loss_reg +=self.sparsityconv(W)
                        if i > 0:
                            self.mu += self.delta_mu
                for j in range(0, len(self.model.classifier)):
                    if isinstance(self.model.classifier[j], nn.Linear):
                        W = self.model.classifier[j].weight
                        B = self.model.classifier[j].bias
                        loss_reg +=self.sparsity(W)
                        self.mu += self.delta_mu

            loss = loss + loss_reg
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            # print(loss.item())
        return epoch_loss/len(data), epoch_acc/len(data)

    def optimize_cont(self,data,ood_class,mask_c, mask_f, nodes, nodes_f, w, w_f,activations_norm, old_model,in_dist_classes):
        epoch_loss = 0
        epoch_acc = 0

        self.model.train()

        for (x,y) in data:
            x = x.to(self.device)
            y = y.to(self.device)

            self.mu = 0
            self.optimizer.zero_grad()
            layers , y_pred = self.model(x)

            y_new = y.clone()
            for j in range(0,len(in_dist_classes)):

                y_new = torch.where(y == in_dist_classes[j], j, y_new)
            y = y_new.clone()


            loss = self.criterion(y_pred,y)

            acc = self.calculate_accuracy(y_pred,y)
            loss_reg = 0
            loss_theta = 0

            if self.l1_reg:
                for name, W in self.model.named_parameters():
                    if name == 'classifier.4.weight':
                        loss_reg += self.alpha*self.L1_regularisation(W)
            #
            idx_layers = 0
            k = 0
            if self.sparsity_flag:

                for i in range(0, len(self.model.features)):
                    if isinstance(self.model.features[i], nn.Conv2d):
                        W = self.model.features[i].weight
                        B = self.model.features[i].bias
                        # loss_reg +=self.sparsityconv(W*torch.Tensor(mask_f[k]).to(self.device))
                        # k+=1
                        loss_reg +=self.sparsityconv(W)
                        W_old = old_model.features[i].weight.detach()
                        B_old = old_model.features[i].bias.detach()
                        W_diff = torch.norm(W - W_old,p=2,dim=(1,2,3))
                        B_diff = torch.abs(B-B_old)
                        activation_tensor = torch.Tensor(activations_norm[idx_layers]).to(self.device)

                        loss_theta += self.beta_weight*torch.sum(activation_tensor*W_diff) + self.beta_bias*torch.sum(activation_tensor*B_diff)
                        # loss_reg +=self.sparsityconv(W)
                        # loss_theta+=self.param_weight*torch.sum(activation_tensor*torch.pow((W_diff**2 + B_diff**2),0.5))

                        idx_layers +=1
                        if i > 0:
                            self.mu += self.delta_mu
                idx_layers +=1
                k=0
                for j in range(0, len(self.model.classifier)):
                    if isinstance(self.model.classifier[j], nn.Linear):
                        W = self.model.classifier[j].weight
                        B = self.model.classifier[j].bias
                        # loss_reg +=self.sparsity(W*torch.Tensor(mask_c[k]).to(self.device))
                        # k+=1
                        loss_reg +=self.sparsity(W)
                        W_old = old_model.classifier[j].weight.detach()
                        B_old = old_model.classifier[j].bias.detach()
                        W_diff = torch.norm(W - W_old,p=2,dim=(1))

                        # B_diff = torch.norm(B-B_old, p=2,dim=(1))
                        B_diff = torch.abs(B-B_old)
                        activation_tensor = torch.Tensor(activations_norm[idx_layers]).to(self.device)

                        loss_theta += self.beta_weight*torch.sum(activation_tensor*W_diff)+self.beta_bias*torch.sum(activation_tensor*B_diff)
                        # loss_theta+=self.param_weight*torch.sum(activation_tensor*torch.pow((W_diff**2 + B_diff**2),0.5))

                        idx_layers +=1
                        self.mu += self.delta_mu


            loss = loss + loss_reg + loss_theta
            loss.backward()



            self.optimizer.step()


            epoch_loss += loss.item()
            epoch_acc += acc.item()
            # pdb.set_trace()
        return epoch_loss/len(data), epoch_acc/len(data)



    def evaluate(self,data,ood_class,in_dist_classes, extract_act = False):

        epoch_loss = 0
        epoch_acc = 0
        y_list = np.ones((self.output_dim))
        activation = np.zeros((self.output_dim,self.penultimate_feats_len))
        average_activation = np.zeros((self.output_dim,self.penultimate_feats_len))
        activation_list = {}
        for i in range(0, self.output_dim):
            activation_list[str(i)] = []

        labels_list = []
        avg_act_all_layers = 0

        self.model.eval()

        with torch.no_grad():

            for (x, y) in data:

                x = x.to(self.device)
                y = y.to(self.device)

                layers ,y_pred = self.model(x)

                if len(ood_class) == 1:
                    y = torch.where(y>ood_class[0], y-1,y)


                elif len(ood_class) > 1:
                    y_new = y.clone()
                    for j in range(0,len(in_dist_classes)):
                        y_new = torch.where(y == in_dist_classes[j], j, y_new)
                    y = y_new.clone()

                loss = self.criterion(y_pred, y)
                acc = self.calculate_accuracy(y_pred, y)
                epoch_loss += loss.item()
                epoch_acc += acc.item()

                if extract_act:
                    idx = y.item()
                    act = layers[len(layers)-2].cpu().numpy()
                    activation[idx] = activation[idx] + act[0,:]
                    y_list[idx] += 1
                        ## append activations
                    activation_list[str(idx)].append(act[0,:])
                    labels_list.append(idx)
            if extract_act:
                for i in range(len(y_list)):
                    average_activation[i] = activation[i]/y_list[i]

        return epoch_loss/len(data), epoch_acc/len(data), average_activation, activation_list, labels_list #,avg_act_all_layers/len(data)


    def ood_evaluate(self,data):

        epoch_loss = 0
        epoch_acc = 0
        activation_list = []
        activations = 0

        self.model.eval()
        with torch.no_grad():

            for (x, y) in data:
                x = x.to(self.device)
                y = y.to(self.device)

                layers ,y_pred= self.model(x)
                top_pred = y_pred.argmax(1, keepdim = True)

                act = layers[len(layers)-2].cpu().numpy()
                activation_list.append(act[0,:])
                activations += act
            average_activation = activations/len(data)
        return epoch_loss/len(data), epoch_acc/len(data),average_activation, activation_list
