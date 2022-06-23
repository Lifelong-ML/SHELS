import numpy as np
import sklearn.metrics as sk
import torch
import pdb
import sys

class_list = np.load('class_list1.npz', allow_pickle = True)['class_list']
save_path = "final_models/MNIST/model1"
if len(sys.argv) > 1:
    save_path = sys.argv[1]
    baseline = sys.argv[2]
# classes = ['top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
# print("JUST COSINE")
print(baseline)
classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four','5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# classes = np.arange(0,10)

# classes = ['00018', '00026', '00000', '00021', '00020', '00010', '00024', '00006', '00001', '00022',
# '00013', '00025', '00041', '00017', '00014', '00035', '00038', '00042', '00002', '00028', '00003',
# '00029', '00027', '00012', '00008', '00034', '00039', '00011', '00005', '00016', '00004', '00032',
#  '00019', '00009', '00015', '00033', '00031', '00040', '00036', '00030', '00023', '00007', '00037']


auroc_list = []
for i in range(0, len(class_list)):
    # i = 9
    curr_list = class_list[i]

    ood_class_list = curr_list[5:10]
    ind_class_list = curr_list[0:5]

    classes_ind = np.array(classes)[ind_class_list]
    ood_classes = np.array(classes)[ood_class_list]
    model = torch.load(save_path+'/model_{}.pt'.format(i), map_location='cpu')
    weights = model['classifier.2.weight'].detach().cpu().numpy()
    weights_norm = np.reshape(np.linalg.norm(weights,axis=1),(-1,1)) + 1e-50
    weights_normalized = weights/weights_norm
    all_test_scores = []

    # activations_train = dict(np.load(save_path+'/activations/act_full_train_{}.npz'.format(i),allow_pickle = True))
    activations_test = dict(np.load(save_path+'/activations/act_full_test_{}.npz'.format(i),allow_pickle = True))
    for k in range(0,len(activations_test)):
        data = activations_test[str(k)]

        #
        if baseline == True:
            data_norm = np.reshape(np.linalg.norm(data,axis=1),(-1,1))
            data_norm = np.where(data_norm == 0, 1e-20, data_norm)
            data_normalized = data/data_norm

            scores = np.matmul(data_normalized, weights_normalized.T)
        else:
            scores = np.matmul(data,weights.T)

        test_scores_list = (np.max(scores,axis = 1)).tolist()
        # print(np.shape(test_scores_list))
        all_test_scores= all_test_scores + test_scores_list
        # print(np.shape(all_test_scores))

    all_ood_scores = []
    # pdb.set_trace()
    # ood_classes = ['top']

    for j in range(0, len(ood_classes)):

        activations_ood = dict(np.load(save_path+'/activations/act_full_ood_{}_{}.npz'.format(ood_classes[j],i), allow_pickle = True))['act_ood']
        # activations_ood = dict(np.load(save_path+'/activations/act_full_ood_{}_{}.npz'.format('top',0), allow_pickle = True))['act_ood']
        # # #
        if baseline == True:
            data_norm = np.reshape(np.linalg.norm(activations_ood,axis=1),(-1,1))
            # pdb.set_trace()
            data_norm = np.where(data_norm == 0, 1e-20, data_norm)
            data_normalized = activations_ood/data_norm

            scores = np.matmul(data_normalized, weights_normalized.T)
        # pdb.set_trace()
        else:
            scores = np.matmul(activations_ood,weights.T)

        ood_scores_list = (np.max(scores,axis = 1)).tolist()
        all_ood_scores = all_ood_scores + ood_scores_list


    labels = np.concatenate([np.zeros_like(all_ood_scores), np.ones_like(all_test_scores)],axis=0)
    scores_all = np.concatenate([all_ood_scores,all_test_scores])
    auroc = sk.roc_auc_score(labels,scores_all)

    auroc_list.append(auroc)
    print( auroc)
print("Mean AUROC", np.mean(auroc_list), np.std(auroc_list))

class_list = np.array([5,12,34,21,8,24,20,40,16,10,18,37,0,27,7,39,28,13],[])
