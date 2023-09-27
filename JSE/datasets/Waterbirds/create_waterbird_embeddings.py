# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:58:18 2022

"""

# torch related libraries
import torch
import torchvision 
from torchvision import transforms
from PIL import Image
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from JSE.helpers import *
from JSE.models import *
import sys
import time
#  standard libraries
import pandas as pd
import numpy as np
import pickle
import os
import argparse

def transform_cub(target_resolution):
  """
    

    Parameters
    ----------
    target_resolution : tuple
        desired (height, width) of image

    Returns
    -------
    transform : function
        to be applied to tensors.Transforms the image.

  """

  scale = 256.0/224.0

  # apply the following transformations
  transform = transforms.Compose([
                                    transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))), # resize
                                    transforms.CenterCrop(target_resolution),  # crop the center
                                    transforms.ToTensor(), # turn to tensor
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #standard transformation for imagenet-trained resnet 50

    ])
  
  return transform



def main(spurious_ratio_values, type_image, model_name, set_confounder_water, spurious_ratio_water, define_tensors, finetuned, adversarial, early_stopping, balance_main, settings, seed_list, device_type='mps'):


  for seed in seed_list:
    # go over each confounder strength
    for spurious_ratio in spurious_ratio_values:

      # select metadata
      print('Working on: {}, seed: {}'.format(spurious_ratio, seed))
      spurious_ratio = str(int(spurious_ratio*100))
      spurious_ratio_water = str(int(spurious_ratio_water*100))

      # set the folder name
      if set_confounder_water:
        folder_name = 'images/waterbirds_'+spurious_ratio + '_confounded_water_' + spurious_ratio_water
      else:
        folder_name = 'images/waterbirds_'+spurious_ratio
      
      # set the metadata filename
      meta_data_filename = folder_name + '/metadata_waterbird_'+spurious_ratio+ '_' +spurious_ratio + '_50'

      # add confounder strength to metadata filename
      if set_confounder_water:
        meta_data_filename += '_confounded_water_' + spurious_ratio_water 
      meta_data_filename += '.csv'
        
      # get the metadata file
      df_images = pd.read_csv(meta_data_filename)

      # gather info per train, validation and test set
      df_images_train = df_images[df_images['split']==0]
      df_images_val = df_images[df_images['split']==1]
      df_images_test = df_images[df_images['split']==2]

      # change to (224, 224) resolution
      resolution_waterbird = (224, 224)

      # define transformationfunction
      tranformation_waterbird = transform_cub(resolution_waterbird)

      # Check whether the specified path exists or not
      tensor_folder = folder_name +'/tensors'
      folder_exists = os.path.exists(tensor_folder)
      if not folder_exists:
            # Create a new directory because it does not exist
            os.makedirs(tensor_folder)

      # check the type
      if define_tensors:
        waterbird_images_combined_train, waterbird_labels_combined_train = load_images(df_images_train, tranformation_waterbird,folder_name+'/', type_image )
        waterbird_images_combined_val, waterbird_labels_combined_val = load_images(df_images_val, tranformation_waterbird,folder_name+'/', type_image )
        waterbird_images_combined_test, waterbird_labels_combined_test = load_images(df_images_test, tranformation_waterbird,folder_name+'/', type_image )


        torch.save(waterbird_images_combined_train, tensor_folder + '/waterbird_images_'+type_image+'_train_' + spurious_ratio+'.pt')
        torch.save(waterbird_labels_combined_train, tensor_folder + '/waterbird_labels_'+type_image+'_train_' + spurious_ratio+'.pt')
        torch.save(waterbird_images_combined_val, tensor_folder + '/waterbird_images_'+type_image+'_val_'  + spurious_ratio+'.pt')
        torch.save(waterbird_labels_combined_val, tensor_folder + '/waterbird_labels_'+type_image+'_val_' + spurious_ratio+'.pt')
        torch.save(waterbird_images_combined_test, tensor_folder + '/waterbird_images_'+type_image+'_test_' + spurious_ratio+'.pt')
        torch.save(waterbird_labels_combined_test, tensor_folder + '/waterbird_labels_'+type_image+'_test_' + spurious_ratio+'.pt')



      if model_name == 'resnet50':
        # load the resnet50 model
        model = torchvision.models.resnet50(pretrained = True)
      elif model_name =='resnet18':
        # load the resnet18 model
        model = torchvision.models.resnet18(pretrained = True)

      device = torch.device(device_type)

      # set the embeddings  
      model_embeddings =  embedding_creator(model)

      if finetuned:
        model_file = 'finetuned_models/{}_model_finetuned_Waterbirds_settings_{}_spurious_ratio_{}_balance_main_{}_early_stopping_{}_adversarial_{}_seed_{}.pt'.format( model_name, settings, spurious_ratio, int(balance_main), int(early_stopping), int(adversarial), seed)
        state_dict = torch.load(model_file)
        model_embeddings.load_state_dict(state_dict)
        print('Loaded finetuned model: {}', model_file)


      # load the tensors - combined images
      waterbird_images_combined_train = torch.load(tensor_folder + '/waterbird_images_{}_train_'.format(type_image) + spurious_ratio+'.pt').to(device)
      waterbird_images_combined_val = torch.load(tensor_folder + '/waterbird_images_{}_val_'.format(type_image)+ spurious_ratio+'.pt').to(device)
      waterbird_images_combined_test = torch.load(tensor_folder + '/waterbird_images_{}_test_'.format(type_image)+ spurious_ratio+'.pt').to(device)

      # put images in tensorDataset
      train_waterbird_combined_images= TensorDataset(waterbird_images_combined_train)
      val_waterbird_combined_images= TensorDataset(waterbird_images_combined_val)
      test_waterbird_combined_images= TensorDataset(waterbird_images_combined_test)

      # loaders for bird dataset
      batch_size=100
      workers=0
      loaders_waterbird = {
          'train_images' : DataLoader(train_waterbird_combined_images, 
                                                batch_size=batch_size, 
                                                shuffle=False, 
                                              num_workers=workers),
          'val_images' : DataLoader(val_waterbird_combined_images, 
                                                batch_size=batch_size, 
                                                shuffle=False),
          'test_images' : DataLoader(test_waterbird_combined_images,
                                                batch_size=batch_size,
                                                shuffle=False)
          }



      # embeddings for combined images
      model_embeddings.to(device)
      waterbird_images_train_embeddings = get_embedding_in_batches(model_embeddings, loaders_waterbird['train_images'])
      waterbird_images_val_embeddings = get_embedding_in_batches(model_embeddings, loaders_waterbird['val_images'])
      waterbird_images_test_embeddings = get_embedding_in_batches(model_embeddings, loaders_waterbird['test_images'])

      # Check whether the specified path exists or not
      folder_embeddings = 'embeddings/waterbirds_'+spurious_ratio + '_' + model_name

      if set_confounder_water:
        folder_embeddings += '_confounded_water_' + spurious_ratio_water
      
      # create the folder if it does not exist
      folder_exists = os.path.exists(folder_embeddings)
      if not folder_exists:
          # Create a new directory because it does not exist
          os.makedirs(folder_embeddings)
      
      embedding_name_train ='/waterbird_images_'+type_image+'_train_'+spurious_ratio+'_embeddings_' + model_name 
      embedding_name_val ='/waterbird_images_'+type_image+'_val_'+spurious_ratio+'_embeddings_' + model_name
      embedding_name_test ='/waterbird_images_'+type_image+'_test_'+spurious_ratio+'_embeddings_' + model_name 

      if finetuned:
        embedding_name_train += '_finetuned'
        embedding_name_val += '_finetuned'
        embedding_name_test += '_finetuned'
      if adversarial:
        embedding_name_train += '_adversarial'
        embedding_name_val += '_adversarial'
        embedding_name_test += '_adversarial'

      if finetuned or adversarial:
        embedding_name_train += '_seed_{}'.format(seed)
        embedding_name_val += '_seed_{}'.format(seed)
        embedding_name_test += '_seed_{}'.format(seed)
        

      torch.save(waterbird_images_train_embeddings, folder_embeddings + embedding_name_train +'.pt')
      torch.save(waterbird_images_val_embeddings, folder_embeddings + embedding_name_val +'.pt')
      torch.save(waterbird_images_test_embeddings, folder_embeddings +embedding_name_test +'.pt')


      # after saving, pause for a minute, then continue
      print('Saved embeddings for spurious ratio: {}'.format(spurious_ratio))
      print('Sleeping for 1 minute')
      time.sleep(60)



if __name__ == "__main__":
    # type_image, model_name, set_confounder_water, spurious_ratio_water, define_tensors):

    parser = argparse.ArgumentParser()
    parser.add_argument('--type_image', help ='type of image')
    parser.add_argument('--model_name', help ='model name')
    parser.add_argument('--spurious_ratio_values', help ='list of confounder strength values')
    parser.add_argument('--set_confounder_water', help ='set confounder water')
    parser.add_argument('--spurious_ratio_water',    help ='confounder strength water')
    parser.add_argument('--define_tensors', help ='define tensors')
    parser.add_argument('--finetuned', help ='finetuned')
    parser.add_argument('--adversarial', help ='adversarial')
    parser.add_argument('--early_stopping', help ='early_stopping')
    parser.add_argument('--balance_main', help ='balance_main')
    parser.add_argument('--settings', help ='settings')
    parser.add_argument('--seed', help ='seed')


    args = parser.parse_args()
    dict_arguments = vars(args)
    spurious_ratio_values = Convert(dict_arguments['spurious_ratio_values'], float)
    set_confounder_water = str_to_bool(dict_arguments['set_confounder_water'])
    spurious_ratio_water = float(dict_arguments['spurious_ratio_water'])
    define_tensors = str_to_bool(dict_arguments['define_tensors'])
    type_image = dict_arguments['type_image']
    model_name = dict_arguments['model_name']
    finetuned = str_to_bool(dict_arguments['finetuned'])
    adversarial = str_to_bool(dict_arguments['adversarial'])
    early_stopping = str_to_bool(dict_arguments['early_stopping'])
    balance_main = str_to_bool(dict_arguments['balance_main'])
    settings = dict_arguments['settings']
    seed = Convert(dict_arguments['seed'], int)

    
    main(spurious_ratio_values, type_image, model_name, set_confounder_water, spurious_ratio_water, define_tensors, finetuned, adversarial, early_stopping, balance_main, settings, seed, device_type='mps')