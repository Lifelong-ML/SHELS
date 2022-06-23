import numpy
import torch

import argparse

def get_args():
    parser = argparse.ArgumentParser()

    ## parser
    parser.add_argument('--dataset1', default="mnist", type=str, required = False)
    parser.add_argument('--dataset2', default="fmnist", type=str, required = False)

    parser.add_argument('--experiment', default="ood_detect", type = str, required = False) ## ood_detect with cont learning?
    parser.add_argument('--multiple_dataset', default = False, type=bool, required = False)

    parser.add_argument('--ID_tasks', default=5, type=int, required = False)
    parser.add_argument('--total_tasks', default=10, type=int, required = False)
    parser.add_argument('--single_ood_class', default = False, type = bool, required = False)  ## in this case does a round robin trough all the classes


    parser.add_argument('--batch_size', default=32, type=int, required = False)
    parser.add_argument('--lr', default=0.0001, type=float, required = False)
    parser.add_argument('--epochs', default = 20, type = int, required = False)
    parser.add_argument('--epochs_g', default = 10, type = int, required = False)

    parser.add_argument('--cosine_sim', default = False, type = bool, required = False)
    parser.add_argument('--baseline', default = False, type = bool,required = False )
    parser.add_argument('--baseline_ood', default = False, type = bool,required = False )
    parser.add_argument('--sparsity_es', default = False, type = bool, required = False)
    parser.add_argument('--sparsity_gs', default = False, type = bool, required = False)

    parser.add_argument('--full_pipeline', default= False, type=bool, required = False)



    parser.add_argument('--train', default = False, type=bool, required = False)
    parser.add_argument('--load_checkpoint', default = False, type = bool, required = False)
    parser.add_argument('--load_list', default = False, type = bool, required = False)
    parser.add_argument('--cont_learner', default = False, type = bool, required = False)
    parser.add_argument('--random_seed', default=5, type=int, required = False)

    parser.add_argument('--save_path', default="test", type = str, required = False)

    args = parser.parse_args()

    return args


def print_args(args):

    print("Experiment :", args.experiment)
    print("Number of In-Dist tasks:", args.ID_tasks)
    print("Cosine Similarity :", args.cosine_sim)
    print("Exclusive Sparsity :", args.sparsity_es)
    print("Group Sparsity :", args.sparsity_gs)
    print("Training", args.train)
    print("dataset1 :", args.dataset1)

    print("learning_rate :", args.lr)
