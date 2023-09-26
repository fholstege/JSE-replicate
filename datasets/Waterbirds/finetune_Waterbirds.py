import os
import sys
import numpy as np
import random
import string
import pandas as pd
from pytorch_revgrad import RevGrad
import torchvision 

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import sys
import math
import sklearn
from twokenize import simpleTokenize
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse
import re
from torch.optim.lr_scheduler import LinearLR

import time
import datetime
from JSE.data import *
from JSE.training import *
from JSE.models import * 
import pickle



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def main(settings, spurious_ratio,balance_main, early_stopping, adversarial, per_step, start_seed, sims, device_type='mps', balance_concept=False):


    if settings == 'kirichenko':
        batch_size = 16
        lr =  (10**-5)
        epochs = 10
        weight_decay = 10**-4
        adv_lambda = 0
    
    elif settings == 'adversarial_param':
        batch_size = 128
        lr =  (10**-3) # same as Sagawa (2020)
        epochs = 20
        weight_decay = (10**-4) # same as Sagawa (2020)
        adv_lambda = 1


    for sim_i in range(sims):

        seed = sim_i + start_seed

        
        # set the device
        device = torch.device(device_type)
        Waterbirds_obj = Waterbird_Dataset()

        # load the images 
        set_seed(seed)
        Waterbirds_obj.load_original_images(spurious_ratio,'combined')

        # if balance_main is True, then we need to get the weights for the main task
        if balance_main:
            weights_train, weights_val = Waterbirds_obj.get_class_weights_train_val(Waterbirds_obj.y_m_train, Waterbirds_obj.y_m_val)
        elif balance_concept:
            weights_train, weights_val = Waterbirds_obj.get_class_weights_train_val(Waterbirds_obj.y_c_train, Waterbirds_obj.y_c_val)
        else:
            weights_train, weights_val = None, None


        # create the loaders
        if adversarial:
            loader_dict = Waterbirds_obj.create_loaders( batch_size, 0, shuffle=True, with_concept=True, include_weights=balance_main, train_weights=weights_train, val_weights=weights_val, concept_first=False)
        else:
            loader_dict = Waterbirds_obj.create_loaders( batch_size, 0, shuffle=True, with_concept=False, include_weights=balance_main, train_weights=weights_train, val_weights=weights_val, concept_first=False)

        # define the resnet50 model from torch
        resnet50 = torchvision.models.resnet50(pretrained = True)
        resnet50_features = embedding_creator(resnet50)
        resnet50_features.to(device)

        # define the model on top of Resnet50
        output_dim = 1
        loss_fn = torch.nn.BCEWithLogitsLoss()
        W = torch.nn.Linear(2048,output_dim , bias=True)
        W.to(device)

        if adversarial:
            print('With Adversarial learning')
            # use RevGrad() module for reversing the gradient, then the classifier
            adv_W =  torch.nn.Sequential( torch.nn.Linear(2048, 1), RevGrad())
            adv_W.to(device)
        else:
            adv_W = None



        if settings == 'kirichenko':

            if adversarial:
                optimizer = torch.optim.AdamW(list(resnet50_features.parameters()) + list(W.parameters()) + list(adv_W.parameters()), lr=lr, weight_decay=weight_decay)
            else: 
                optimizer = torch.optim.AdamW(list(resnet50_features.parameters()) + list(W.parameters()), lr=lr, weight_decay=weight_decay)
            
            use_scheduler = True
            scheduler = 'linear'

        elif settings == 'adversarial_param':

            if adversarial:
                optimizer = torch.optim.SGD(list(resnet50_features.parameters()) + list(W.parameters()) + list(adv_W.parameters()), momentum = 0.9, lr=lr, weight_decay=weight_decay)
            else: 
                optimizer = torch.optim.SGD(list(resnet50_features.parameters()) + list(W.parameters()), momentum = 0.9, lr=lr, weight_decay=weight_decay)

            use_scheduler = False
            scheduler = None
            
    


        baseline_acc_concept = Waterbirds_obj.y_c_train.mean()
        print('Baseline accuracy concept: {}'.format(baseline_acc_concept))
        set_seed(seed)
        train_resnet50(epochs, resnet50_features, W, loader_dict, 'train', 'val', device, optimizer, per_step=per_step, early_stopping= early_stopping, orig_patience=1, tol = 0.0001, use_scheduler=use_scheduler, scheduler=scheduler, save_best_model=True, model_name='resnet50_finetune', loss_fn =  loss_fn ,  adversarial=adversarial, adv_W=adv_W, baseline_acc_concept=baseline_acc_concept, adv_lambda=adv_lambda)   
        torch.save(resnet50_features.state_dict(), 'finetuned_models/resnet50_model_finetuned_Waterbirds_settings_{}_spurious_ratio_{}_balance_main_{}_early_stopping_{}_adversarial_{}_seed_{}.pt'.format(settings, int(100*spurious_ratio), int(balance_main), int(early_stopping), int(adversarial), seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings')
    parser.add_argument('--spurious_ratio')
    parser.add_argument('--early_stopping')
    parser.add_argument('--adversarial')
    parser.add_argument('--balance_main')
    parser.add_argument('--per_step')
    parser.add_argument('--seed')
    parser.add_argument('--sims')
   
    args = parser.parse_args()

    
    settings = args.settings
    spurious_ratio = float(args.spurious_ratio)
    early_stopping = str_to_bool(args.early_stopping)
    adversarial = str_to_bool(args.adversarial)
    balance_main = str_to_bool(args.balance_main)
    per_step = int(args.per_step)
    seed = int(args.seed)
    sims = int(args.sims)

    main(settings, spurious_ratio,balance_main, early_stopping, adversarial,per_step, seed, sims)


