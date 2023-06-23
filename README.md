# SHELS: Exclusive Feature Sets For Novelty Detection And Continual Learning Without Class Boundaires

 ## 1. Requirements
  - pip install -r requirements.txt
   
    or 
  - conda create --name <env_name> --file requirements.txt


## 2. Datsets

  - Pytorch datasets for MNIST, FMNIST, SVHN and CIFAR10
  - GTSRB can be downloaded here https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

    Make sure to set the data_path variable in main.py to the data folder path

## 3. Novelty detection 
   create directories to save the models and activations, example mkdir dir dir_bl
  ### MNIST (within-datatset)
  #### train 
    ### shels
     python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 5 --save_path ./dir

    ### baseline
     python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --baseline True --train True --random_seed 5 --save_path ./dir_bl

   #### evaluation
     ### shels
      python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 5 --save_path ./dir

     ### baseline
      python main.py --dataset1 mnist --ID_tasks 5 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --baseline True --load_checkpoint True --random_seed 5 --save_path ./dir_bl --baseline_ood True

  To run experiments with different datasets, choose dataset1 argument from [mnist, fmnist, cifar10, svhn, gtsrb].
  
  Note : Be sure to specify the --total_tasks as well as --ID_tasks arguments, total number of classes and total number of ID classes respectively

  ### MNIST (ID) vs FMNIST (OOD) (across-datasets)
   #### train
    ### shels
     python main.py --dataset1 mnist --dataset2 fmnist --multiple_dataset True --ID_tasks 10 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --save_path ./dir

    ### baseline
     python main.py --dataset1 mnist --dataset2 fmnist --multiple_dataset True --ID_tasks 10 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --baseline True --train True --save_path ./dir_bl

   ### evaluation
    ### shels
     python main.py --dataset1 mnist --dataset2 fmnist --multiple_dataset True --ID_tasks 10 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --save_path ./dir

    ### baseline
     python main.py --dataset1 mnist --dataset2 fmnist --multiple_dataset True --ID_tasks 10 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --baseline True --load_checkpoint True --save_path ./dir_bl --baseline_ood True

 To run experiments with different datasets, choose dataset1 and dataset2 from [mnist, fmnist, cifar10, svhn, gtsrb]
    
  Note : Be sure to specify the --total_tasks as well as --ID_tasks arguments and ensure cosistent input dimension in data_loader.py for ID and OOD datasets

## 4. Novelty Accommodation and Novelty detection-Accommodation 
   
   ### MNIST
   #### train 
     python main.py --dataset1 mnist --ID_tasks 7 --total_tasks 10 --batch_size 32 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --train True --random_seed 5 --save_path ./dir

   #### accommodation
    python main.py --dataset1 mnist --ID_tasks 7 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 5 --save_path ./dir --cont_learner True


   #### detection and accommodation
    python main.py --dataset1 mnist --ID_tasks 7 --total_tasks 10 --batch_size 1 --lr 0.0001 --epochs 10 --cosine_sim True --sparsity_gs True --load_checkpoint True --random_seed 5 --full_pipeline True --save_path ./dir 


  To run experiments with different datasets, choose dataset1 from [mnist, fmnist, cifar10, svhn, gtsrb]
  To load a preloaded set of experiments, use class_list_GTSRB.npz for GTSRB dataset and class_list1.npz for the other datasets by setting--load_list flag to True.
  
## Citing this work
  If you use this work please cite our paper.
      
  
 ```
@inproceedings{gummadi2022shels
  title = {SHELS: Exclusive Feature Sets for Novelty Detection and Continual Learning Without Class Boundaries},
  authors = {Gummadi, Meghna and Kent, David and Mendez, Jorge A. and Eaton, Eric},
  booktitle = {1st Conference on Lifelong Learning Agents (CoLLAs-22)},
  year = {2022}
}
```

