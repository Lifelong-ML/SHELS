import numpy as np
import torch
import torch.nn as nn

import pdb

def get_weights_mask(model, layer_act, layer_idx, output_dim):

    nodes_list_feats = []
    mask_list_feats = []
    weights_list_feats = []
    ## initialize nodes
    lower_nodes = np.zeros(1)
    curr_idx = 0
    activations_norm = []
    with torch.no_grad():

        for i in range(0,len(model.features)):

            if isinstance(model.features[i], nn.Conv2d):

                weights = model.features[i].weight.data.clone().detach()
                upper_nodes = np.zeros(model.features[i].out_channels)

                activations = layer_act[layer_idx[curr_idx]]
                curr_idx+=1
                # print(activations.shape)
                if len(activations.shape)>1:
                    activations = activations.norm(p = 2, dim=(1,2)).cpu().numpy()
                    activations_norm.append(activations)
                    # print(activations.shape)
                ## lets freeze all weights
                mask = np.ones((weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]))
                ## fix the outgoing weights of lower unimporant nodes to 0
                non_imp_idx_lower = np.argwhere(lower_nodes != 0)
                imp_idx_lower = np.argwhere(lower_nodes == 0)
                # if len(non_imp_idx_lower) > 0:
                #     weights[:,non_imp_idx_lower,:,:] = 0.0

                ## now lets look for unimportant weights in the higher nodes and unfreeze the weights coming into it
                non_imp_idx_upper = np.argwhere(activations == 0)[:,0]
                imp_idx_upper = np.argwhere(activations != 0)[:,0]
                upper_nodes[non_imp_idx_upper] = 1
                mask[imp_idx_upper,imp_idx_lower,:,:] = 0
                # mask[:,:,:] = 0
                if len(non_imp_idx_lower) > 0:
                    weights[imp_idx_upper,non_imp_idx_lower,:,:] = 0.0
                #weights[non_imp_idx_upper,non_imp_idx_lower,:] = torch.randn(weights[non_imp_idx_upper,non_imp_idx_lower,:].size()).to("cuda:0")*.001
                # torch.nn.init.xavier_uniform_(weights[non_imp_idx_upper,non_imp_idx_lower,:,:] , gain=1.0)
                # print("total unimportant higher nodes :", len(non_imp_idx_upper))

                nodes_list_feats.append(upper_nodes)
                lower_nodes = upper_nodes
                mask_list_feats.append(mask)
                weights_list_feats.append(weights)


        ## nn.flatten activations
        activations = layer_act[layer_idx[curr_idx]].cpu().numpy()
        activations_norm.append(activations)
        lower_nodes = np.zeros(len(activations))
        non_imp_idx_lower = np.argwhere(activations == 0)[:,0]
        lower_nodes[non_imp_idx_lower] = 1
        # pdb.set_trace()
        curr_idx+=1
        nodes_list = []
        weights_list = []
        mask_list = []

        for i in range(0,len(model.classifier)):

            if isinstance(model.classifier[i], nn.Linear):

                weights = model.classifier[i].weight.data.clone().detach()
                upper_nodes = np.zeros(model.classifier[i].out_features)
                # pdb.set_trace()
                if i < 2:
                    activations = layer_act[layer_idx[curr_idx]].cpu().numpy()
                    activations_norm.append(activations)
                    curr_idx+=1
                    # print(activations.shape)
                else:

                    activations = np.ones(output_dim)
                    activations[output_dim -1] = 0
                    activations_norm.append(activations)

                ## lets freeze all weights
                mask = np.ones((weights.shape[0], weights.shape[1]))
                ## fix the outgoing weights of lower unimporant nodes to 0
                non_imp_idx_lower = np.argwhere(lower_nodes == 1)
                imp_idx_lower = np.argwhere(lower_nodes == 0)
                # if len(non_imp_idx_lower) > 0:
                #     weights[:,non_imp_idx_lower] = 0.0

                ## now lets look for unimportant weights in the higher nodes and unfreeze the weights coming into it
                imp_idx_upper = np.argwhere(activations != 0)[:,0]
                non_imp_idx_upper = np.argwhere(activations == 0)[:,0]
                upper_nodes[non_imp_idx_upper] = 1
                mask[imp_idx_upper, imp_idx_lower] = 0
                if len(non_imp_idx_lower) > 0:
                    weights[imp_idx_upper,non_imp_idx_lower] = 0.0


                # weights[non_imp_idx_upper,non_imp_idx_lower] = torch.randn(weights[non_imp_idx_upper,non_imp_idx_lower].size()).to("cuda:0")*.001
                # torch.nn.init.xavier_uniform_(weights[non_imp_idx_upper,non_imp_idx_lower] , gain=1.)
                # print("total unimportant higher nodes :", len(non_imp_idx_upper))

                nodes_list.append(upper_nodes)
                lower_nodes = upper_nodes
                mask_list.append(mask)
                weights_list.append(weights)




    return mask_list, weights_list, nodes_list, mask_list_feats, weights_list_feats, nodes_list_feats, activations_norm
