import os
import sys
import numpy as np
import random
import string
import pandas as pd
from pytorch_revgrad import RevGrad

from transformers import BertModel, BertTokenizer, BertForSequenceClassification
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
import pickle


# Function to calculate the accuracy of our predictions vs labels
def get_acc_BERT(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def main(settings, finetune_mode, sample_data, train_size, val_size, test_size, spurious_ratio, balance_main, start_seed, sims, device_type='mps', dropout=0.1, adversarial=False, early_stopping=True, use_punctuation=False, binary_task=True):

    if settings == 'kumar':
        # parameters from Kumar et al. (2022) - based on mail correspondence
        batch_size = 32
        lr =5 * (10**-5)
        epochs = 5
        adv_lambda = 0

    elif settings == 'kirichenko':
        batch_size = 16
        lr =  (10**-5)
        epochs = 10
        weight_decay = 10**-4
        adv_lambda = 0
    
    elif settings == 'kirichenko_adapted':
        batch_size = 16
        lr =  (10**-5)
        epochs = 5
        weight_decay = 10**-4
        adv_lambda = 0
    
    elif settings == 'RLACE':
        batch_size = 10
        lr = 0.0005
        epochs = 5
        weight_decay = (10**-6)
        adv_lambda = 0

    elif settings == 'adversarial_param':
        batch_size = 32
        lr =  (10**-5)
        epochs = 10
        weight_decay = 10**-4 
        adv_lambda = 1

    elif settings == 'standard_multiNLI':
        batch_size = 32
        lr =  (10**-5)
        epochs = 10
        weight_decay = 0
        adv_lambda = 0


       

    for sim_i in range(sims):
            
        seed = sim_i + start_seed
        set_seed(seed)
    
            

        device = torch.device(device_type)
        multiNLI_obj = multiNLI_Dataset()

        if not use_punctuation:
            multiNLI_obj.load_main_dataset('raw_input/multinli_bert_features', 'processed', turn_binary=binary_task)
        else:
            multiNLI_obj.load_main_dataset('raw_input/multinli_bert_features_punctuated', 'raw_input/multinli_bert_features_punctuated', turn_binary=binary_task, punctuation=True)
    

        if not sample_data:
            sample = False
        else:
            sample = True
            multiNLI_obj.set_sample_subset(train_size, val_size, test_size, spurious_ratio, 0.5, seed=seed)

        # load the train and val sample ids from dict
        loader_dict = multiNLI_obj.create_loaders_for_BERT(batch_size=batch_size,workers=0, sample=sample, adversarial=adversarial)

        # define the model
        bert = BertModel.from_pretrained(  "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
                                                        output_attentions = False, # Whether the model returns attentions weights.
                                                        output_hidden_states = False)# Whether the
        bert.to(device)

        # define the model on top of BERT
        if binary_task:
            output_dim = 1
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            output_dim = 3
            loss_fn = torch.nn.CrossEntropyLoss()

        
            
        W = torch.nn.Linear(768,output_dim , bias=True)
        W.to(device)

        if adversarial:
            print('adversarial learning')

            # use RevGrad() module for reversing the gradient, then the classifier
            adv_W =  torch.nn.Sequential(torch.nn.Linear(768, 1),  RevGrad()
                                       ) 
            adv_W.to(device)
        else:
            adv_W = None


        # define the optimizer

        if settings == 'kumar' or settings == 'standard_multiNLI' or settings == 'adversarial_param':

            if adversarial:
                optimizer = torch.optim.Adam(list(bert.parameters()) + list(W.parameters()) + list(adv_W.parameters()), lr=lr)
            else:
                optimizer = torch.optim.Adam(list(bert.parameters()) + list(W.parameters()), lr=lr)

            use_scheduler = False
            scheduler = None


        elif settings == 'kirichenko' or settings == 'kirichenko_adapted':

            if adversarial:
                optimizer = torch.optim.AdamW(list(bert.parameters()) + list(W.parameters()) + list(adv_W.parameters()), lr=lr, weight_decay=weight_decay)
            else: 
                optimizer = torch.optim.AdamW(list(bert.parameters()) + list(W.parameters()), lr=lr, weight_decay=weight_decay)

            if adversarial:
                use_scheduler = False
                scheduler = None
            else:
                use_scheduler = True
                scheduler = 'linear'
          
        elif settings == 'RLACE':

            if adversarial:
                optimizer = torch.optim.SGD(list(bert.parameters()) + list(W.parameters()) + list(adv_W.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)
            else:
                optimizer = torch.optim.SGD(list(bert.parameters()) + list(W.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)

            use_scheduler = False
            scheduler = None

        print('Finetune mode is: {}'.format(finetune_mode))

        if dropout != 0:
            dropout_layer = torch.nn.Dropout(p=dropout)
        else:
            dropout_layer = None

        set_seed(seed)
        train_BERT(epochs, finetune_mode, bert, W, loader_dict, 'train', 'val', device, optimizer, dropout=dropout_layer, per_step=1, early_stopping= early_stopping, orig_patience=1, tol = 0.0001, use_scheduler=use_scheduler, scheduler=scheduler, save_best_model=True, model_name='BERT_finetune_multiNLI', loss_fn =  loss_fn ,  adversarial=adversarial, adv_W=adv_W, adv_lambda=adv_lambda)   

        if not sample_data:
            train_size = 'all'
            val_size = 'all'

        filename_model =  'finetuned_models/bert_model_finetuned_MNLI_settings_{}_train_size_{}_val_size_{}_spurious_ratio_{}_binary_{}_dropout_{}_early_stopping_{}_finetune_mode_{}_seed_{}'.format( settings, train_size, val_size, int(100*spurious_ratio), int(binary_task), int(100*dropout), int(early_stopping), finetune_mode, seed)

        if use_punctuation:
            filename_model += '_punctuation'

        if adversarial:
            filename_model += '_adversarial'
        filename_model += '.pt'

        torch.save(bert.state_dict(), filename_model)

        if sample_data:
            # create dict with sample ids
            dict_sample_ids = {'train_sample': multiNLI_obj.sample_train_ids, 'val_sample': multiNLI_obj.sample_val_ids, 'test_sample': multiNLI_obj.sample_test_ids}

            # save the dict as pkl
            filename = 'sample_ids_MNLI_settings_{}_train_size_{}_val_size_{}_spurious_ratio_{}_binary_{}_dropout_{}_early_stopping_{}_finetune_mode_{}_seed_{}'.format( settings, train_size, val_size, int(100*spurious_ratio), int(binary_task),int(100*dropout), int(early_stopping), finetune_mode, seed)

            if use_punctuation:
                filename += '_punctuation'
            filename += '.pkl'
            with open('finetuned_models/' + filename, 'wb') as f:
                pickle.dump(dict_sample_ids, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings')
    parser.add_argument('--finetune_mode')
    parser.add_argument('--sample_data')
    parser.add_argument('--train_size')
    parser.add_argument('--val_size')
    parser.add_argument('--test_size')
    parser.add_argument('--spurious_ratio')
    parser.add_argument('--dropout')
    parser.add_argument('--early_stopping')
    parser.add_argument('--adversarial')
    parser.add_argument('--balance_main')
    parser.add_argument('--use_punctuation')
    parser.add_argument('--seed')
    parser.add_argument('--sims')
   
    args = parser.parse_args()

    
    settings = args.settings
    finetune_mode = (args.finetune_mode)
    sample_data = str_to_bool(args.sample_data)
    train_size = int(args.train_size)
    val_size = int(args.val_size)
    test_size = int(args.test_size)
    spurious_ratio = float(args.spurious_ratio)
    dropout = float(args.dropout)
    early_stopping = str_to_bool(args.early_stopping)
    adversarial = str_to_bool(args.adversarial)
    balance_main = str_to_bool(args.balance_main)
    use_punctuation = str_to_bool(args.use_punctuation)
    seed = int(args.seed)
    sims = int(args.sims)

    main(settings, finetune_mode,  sample_data, train_size, val_size, test_size, spurious_ratio,balance_main, seed, sims, dropout=dropout, early_stopping=early_stopping, adversarial=adversarial, use_punctuation=use_punctuation)


