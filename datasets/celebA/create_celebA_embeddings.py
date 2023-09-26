
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:06:08 2022

@author: flori
"""

# standard libraries 
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle

# for images
from PIL import Image
from tqdm import tqdm
import cv2

# torch related
import torch
import torchvision 
from torchvision import transforms
from PIL import Image
from torch import nn

from JSE.helpers import *
from JSE.data import * 

import argparse
import os
from JSE.settings import data_info


def main(dataset_setting, sample_n_images, spurious_ratio, seed_list, finetuned=True, settings='adversarial_param', adversarial=True, early_stopping=True, device_type='mps'):


    

    for seed in seed_list:
        # create data object
        data_obj = celebA_Dataset()

        # set the device
        device = torch.device(device_type)
        celebA_obj = celebA_Dataset()

        dataset_settings = dataset_settings = data_info['celebA'][dataset_setting]
            
            

        # define the transform function 
        original_resolution = (178, 178)
        transform_func = transforms.Compose([
                                        transforms.CenterCrop(original_resolution),  # crop the center
                                        transforms.ToTensor(), # turn to tensor
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])] #standard transformation for imagenet-trained resnet 50
                                                    
                                       )
        train_size = dataset_settings['train_size']
        val_size = dataset_settings['val_size']
        test_size = dataset_settings['test_size']
        total_size = train_size + val_size + test_size

        main_task_name = dataset_settings['main_task_name']
        concept_name = dataset_settings['concept_name']


     

        celebA_obj.load_dataset( transform_func, 
                                main_task_name=main_task_name, 
                                concept_name=concept_name, 
                                sample_n_images=total_size,
                                  spurious_ratio = spurious_ratio, 
                                  train_size=train_size, 
                                  val_size=val_size,
                                  test_size=test_size,
                                seed=seed, 
                                load_full_images=True, 
                                create_embeddings=True, 
                                folder='raw_input/',
                                  device_type='mps',
                                   finetuned=finetuned,
                                    settings=settings,
                                    early_stopping=early_stopping,
                                    adversarial=adversarial,
                                    device=device,
                                    balance_main=False,
                                  )



        # save a dict with X_train, y_c_train, y_m_train, X_val, etc. 
        data_dict = {'X_train': celebA_obj.X_train,
                    'X_val': celebA_obj.X_val,
                    'X_test':celebA_obj.X_test,
                    'y_c_train': celebA_obj.y_c_train,
                    'y_c_val': celebA_obj.y_c_val,
                    'y_c_test': celebA_obj.y_c_test,
                    'y_m_train': celebA_obj.y_m_train,
                    'y_m_val':celebA_obj.y_m_val,
                    'y_m_test':celebA_obj.y_m_test,
                    'df_metadata_train': celebA_obj.df_metadata_train,
                    'df_metadata_val': celebA_obj.df_metadata_val,
                    'df_metadata_test': celebA_obj.df_metadata_test,
                    }
        
        # define folder name, check if exists
        sample_n_images_str = 'all' if sample_n_images is None else str(sample_n_images)
        folder_name = 'embeddings'

        if finetuned:
            file_name = 'celebA_main_task_' + main_task_name + '_concept_' + concept_name + '_sample_n_images_' +sample_n_images_str + '_finetuned_settings_' + settings + '_spurious_ratio_' + str(int(spurious_ratio*100))+ '_early_stopping_' + str(int(early_stopping))+  '_adversarial_'  + str(int(adversarial)) + '_seed_' + str(seed) + '.pickle'
        else:
            file_name = 'celebA_main_task_' + main_task_name + '_concept_' + concept_name + '_sample_n_images_' +sample_n_images_str + '.pickle'

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # save as pickle file at folder name
        with open(folder_name   +'/' + file_name, 'wb') as fp:
            pickle.dump(data_dict, fp)

    



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_setting',type=str, help ='dataset setting')
    parser.add_argument('--spurious_ratio',type=float, help ='list of spurious ratio values')
    parser.add_argument('--sample_n_images',type=str, help ='number of images to sample')
    parser.add_argument('--seed',type=str, help ='list of seeds')
    parser.add_argument('--finetuned',type=str, default='False',  help ='whether to use finetuned model')
    parser.add_argument('--settings',type=str, default='adversarial_param', help ='settings for finetuned model')
    parser.add_argument('--early_stopping',type=str, default='True', help ='whether to use early stopping')
    parser.add_argument('--adversarial',type=str, default='True', help ='whether to use adversarial training')
    parser.add_argument('--device_type',type=str, default='mps', help ='device type')

    args = parser.parse_args()
    dict_arguments = vars(args)

    dataset_setting = dict_arguments['dataset_setting']
    sample_n_images = dict_arguments['sample_n_images']
    spurious_ratio = dict_arguments['spurious_ratio']
    seed_list = Convert(dict_arguments['seed'], int)
    finetuned = str_to_bool(dict_arguments['finetuned'])
    settings = dict_arguments['settings']
    early_stopping = str_to_bool(dict_arguments['early_stopping'])
    adversarial = str_to_bool(dict_arguments['adversarial'])
    device_type = dict_arguments['device_type']

   
    main(dataset_setting,  sample_n_images, spurious_ratio, seed_list, finetuned=finetuned, settings=settings, adversarial=adversarial, early_stopping=early_stopping, device_type=device_type)
