import numpy as np
import torch

from scipy.spatial import distance
from sklearn.metrics import jaccard_score
import pdb
from tabulate import tabulate
import sys
if __name__=='__main__':


    class_list = np.load('../class_list1.npz', allow_pickle = True)['class_list']
    # save_path = 'excl_models/excl_fmnist'
    save_path = './dir'
    if len(sys.argv) > 1:
        save_path = sys.argv[1]
    # classes = ['top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    # print("JUST COSINE")
    classes = np.arange(0,10)
    total_mean = []
    for i in range(0, len(class_list)):
        # i = 9
        curr_list = class_list[i]
        ood_class_list = curr_list[5:10]
        ind_class_list = curr_list[0:5]

        classes_ind = np.array(classes)[ind_class_list]
        ood_classes = np.array(classes)[ood_class_list]

        activations_train = dict(np.load(save_path+'/activations/act_full_train_{}.npz'.format(i),allow_pickle = True))
        activations_test = dict(np.load(save_path+'/activations/act_full_test_{}.npz'.format(0),allow_pickle = True))
        table = []
        headers = classes_ind.tolist()
        headers.insert(0,"classes :")
        headers.append('average')
        model = torch.load(save_path+'/model_{}.pt'.format(i))
        weights = model['classifier.2.weight'].detach().cpu().numpy()

        # print(tabulate(table, headers, tablefmt="plain"))


        for idx1 in range(0, len(classes_ind)):
            # print(classes_ind[idx1])
            row = [classes_ind[idx1]]
            # row = [ood_classes[idx1]]
            # activations_ood = dict(np.load(save_path+'/activations/act_full_ood_{}_{}.npz'.format(ood_classes[idx1],i), allow_pickle = True))['act_ood']
            average_activation_1 = np.mean(activations_train[str(idx1)], axis=0)*weights[idx1,:]
            # average_activation_1 = weights[idx1,:]
            # average_activation_1 = np.mean(activations_ood, axis=0)
            encoding_1 = np.where(average_activation_1 > 0 , 1, 0)


            for idx2 in range(0, len(classes_ind)):
                average_activation_2 = np.mean(activations_train[str(idx2)], axis=0)*weights[idx2,:]
                # average_activation_2 = weights[idx2,:]
                encoding_2 = np.where(average_activation_2 > 0, 1, 0)
                # print("Number of nonzeros", np.count_nonzero(encoding_1), np.count_nonzero(encoding_2))

                # exclsive_metric = distance.cosine(encoding_1, encoding_2)#/np.max([np.count_nonzero(encoding_1), np.count_nonzero(encoding_2)])

                # exclsive_metric = 1 - jaccard_score(encoding_1,encoding_2)
                metric_TT = np.sum(np.logical_and(encoding_1, encoding_2)*1)
                metric_TF = np.sum(np.logical_xor(encoding_1, encoding_2)*1)
                metric_FF = np.sum(~np.logical_and(encoding_1, encoding_2)*1)
                # # # exclsive_metric = np.sum(np.logical_xor(encoding_1, encoding_2)*1)/np.max([np.count_nonzero(encoding_1), np.count_nonzero(encoding_2)])
                exclsive_metric = metric_TF/(metric_TF + metric_TT)
                    # print(f'Exclusivity between {classes_ind[idx1]} and {classes_ind[idx2]} : {exclsive_metric}')
                row.append(exclsive_metric)
            mean_per_row = np.sum(row[1:6])/4
            row.append(mean_per_row)

                # print(row)
            table.append(row)
        table_array = np.array(table, dtype = object)
        mean_per_table = np.mean(table_array[:,6])
        print(tabulate(table, headers, tablefmt="plain"))
        print("TABLE AVERAGE :", mean_per_table)
        total_mean.append(mean_per_table)
        print("______________________________________________")
    print(np.mean(total_mean), np.std(total_mean))
