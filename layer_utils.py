import numpy as np
import torch
import torch.nn as nn
from train import Trainer

import pdb

def return_all_layer_activations(trainer, data):
    layer_indices = []
    all_layer_activations = []
    no_batches = 1

    for i in range(0, len(trainer.model.features)):
        if isinstance(trainer.model.features[i], nn.ReLU):
            layer_indices.append(i)
    layer_indices.append(i)
    for j in range(0, len(trainer.model.classifier)):
        if isinstance(trainer.model.classifier[j], nn.ReLU):
            layer_indices.append(i+j+1)



    avg_act_all_layers = 0
    total_batches = 0
    trainer.model.eval()

    with torch.no_grad():
        for (x,y) in data:
            total_batches +=1
            x = x.to(trainer.device)
            y = y.to(trainer.device)

            layers ,y_pred = trainer.model(x)
            avg_layers = []
            for i in range(0, len(layers)):

                mean_layer = torch.mean(layers[i], dim = 0)

                avg_layers.append(mean_layer)

            avg_act_all_layers+=np.array(avg_layers)

    all_layer_activations = avg_act_all_layers/total_batches

    return all_layer_activations, layer_indices
