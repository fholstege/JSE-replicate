
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:06:08 2022

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
from pycocotools.mask import encode, decode
import cv2

from JSE.data_generation_functions import *
from JSE.base_functions import *
import argparse
import os



#val_frac = 0.2
#confounder_strength_values = [0.9]
#set_confounder_water = True
#confounder_strength_water = 0.9

def main(confounder_strength_values, set_confounder_water, confounder_strength_water, val_frac=0.2):

    # here we store the place data 
    place_dir = 'raw_input/water_land'

    for confounder_strength in confounder_strength_values:
        print('Working on: {}'.format(confounder_strength))
        # specify the filename metadata
        if set_confounder_water:
            filename_metadata = 'metadata_waterbird_{}_{}_50_confounded_water_{}.csv'.format(str(int(confounder_strength*100)), str(int(confounder_strength*100)), str(int(confounder_strength_water*100)))
        else:
            filename_metadata = 'metadata_waterbird_{}_{}_50.csv'.format(str(int(confounder_strength*100)), str(int(confounder_strength*100)))

        # specify the folder name

        if set_confounder_water:
            folder_name = 'waterbirds_'+str(int(confounder_strength*100))+'_confounded_water_'+str(int(confounder_strength_water*100))
        else:
            folder_name = 'waterbirds_'+str(int(confounder_strength*100))

        # Check whether the specified path exists or not
        folder_exists = os.path.exists('images/'+ folder_name)
        if not folder_exists:
            # Create a new directory because it does not exist
            os.makedirs('images/'+ folder_name)

        # specify the output folder
        output_dir = 'images/'+folder_name


        ##################
        # 1. Load in the meta-data on the CUB dataset, 
        # determine which birds are waterbirds and others are landbirds
        ##################


        # get overview of all the files
        image_text_file_path = 'images.txt'
        df_images = pd.read_csv(image_text_file_path, # path to images 
                                    sep = " ", # separate the files
                                    header = None, # no header
                                    names = ['img_id', 'img_filename'], # set column names
                                    index_col = 'img_id')

        # get a list of all the bird species
        all_bird_species = [img_filename.split('/')[0].split('.')[1].lower() for img_filename in df_images['img_filename']]

        # get a unique list of all the bird species
        unique_bird_species = np.unique(all_bird_species)

        # all the species that are waterbirds
        water_birds_list = [
            'Albatross', # Seabirds
            'Auklet',
            'Cormorant',
            'Frigatebird',
            'Fulmar',
            'Gull',
            'Jaeger',
            'Kittiwake',
            'Pelican',
            'Puffin',
            'Tern',
            'Gadwall', # Waterfowl
            'Grebe',
            'Mallard',
            'Merganser',
            'Guillemot',
            'Pacific_Loon'
        ]

        # dict that saves per species if waterbird or not
        water_birds = {}

        # go over each species
        for species in unique_bird_species:
            water_birds[species] = 0 # standard; 0 (not water bird)
            for water_bird in water_birds_list: # go over the water birds
                if water_bird.lower() in species: 
                    water_birds[species] = 1 # set if water bird in species

        # add variable with 1 if water bird, 0 if not water bird
        df_images['y'] = [water_birds[species] for species in all_bird_species]


        ##################
        # 2. Determine the train, validation and test split
        # and the percentage of backgrounds for each
        ##################

        # save a dataframe with the training and test split
        train_test_df =  pd.read_csv(
        'train_test_split.txt',
            sep=" ",
            header=None,
            names=['img_id', 'split'],
            index_col='img_id')

        # add column with image id added to it
        df_images = df_images.join(train_test_df, on='img_id')

        # acquire test, train and validation id
        test_ids = df_images.loc[df_images['split'] == 0].index
        train_ids = np.array(df_images.loc[df_images['split'] == 1].index)
        val_ids = np.random.choice(
            train_ids,
            size=int(np.round(val_frac * len(train_ids))),
            replace=False)

        # set the split id
        df_images.loc[train_ids, 'split'] = 0
        df_images.loc[val_ids, 'split'] = 1
        df_images.loc[test_ids, 'split'] = 2

        # standard value of place is zero
        df_images['place'] = 0

        if set_confounder_water:
            df_images['place_type'] = 0

        # train, validation and test ids
        train_ids = np.array(df_images.loc[df_images['split'] == 0].index)
        val_ids = np.array(df_images.loc[df_images['split'] == 1].index)
        test_ids = np.array(df_images.loc[df_images['split'] == 2].index)

        # go over (1) type of split and (2) ids in that type of split
        for split_idx, ids in enumerate([train_ids, val_ids, test_ids]):
            for y in (0, 1): # go over cases; waterbird, landbird
            
                if split_idx == 0 or split_idx == 1: # train and validation
                
                    # set likelihood of appearing in corresponding background
                    if y == 0:
                        pos_fraction = 1 - confounder_strength # if land bird, 1- confounder_strength chance of having a water background
                    else:
                        pos_fraction = confounder_strength # if waterd bird, confounder_strength chance of having a water background

                
                # if test set (split_idx == 2), 50/50
                else:
                    pos_fraction = 0.5

                    
                
                # df for the split
                subset_df = df_images.loc[ids, :]
                
                # y values for this split
                y_ids = np.array((subset_df.loc[subset_df['y'] == y]).index)
                
                # ids of position place
                pos_place_ids = np.random.choice(
                    y_ids,
                    size=int(np.round(pos_fraction * len(y_ids))),
                    replace=False)
                
                # set the ids where place is 1
                df_images.loc[pos_place_ids, 'place'] = 1

            

        if set_confounder_water:
                    
            for split_idx, ids in enumerate([train_ids, val_ids, test_ids]):

                for y_m in (0, 1):
                    for y_c in (0, 1): 

                        if split_idx == 0 or split_idx == 1:

                            if y_c ==0: # if dealing with land bird
                                place_type_frac = 0.5  # type is random
                            else: # if dealing with water bird
                                if y_m == 1:
                                    place_type_frac = confounder_strength_water # type is water with 1 - confounder_strength_water chance
                                else:
                                    place_type_frac = 1 - confounder_strength_water
                            
                        # if we set a confounder in the water background
                        else:
                            place_type_frac = 0.5

                        # df for the split
                        subset_df = df_images.loc[ids, :]
                        subset_df_y_m_y = subset_df.loc[subset_df['y'] == y_m]

                        # y values for this split
                        y_c_ids = np.array((subset_df_y_m_y.loc[subset_df_y_m_y['place'] == y_c]).index)

                        # if set confounder water
                        pos_place_type = np.random.choice(
                            y_c_ids,
                            size =int(np.round(place_type_frac * len(y_c_ids))),
                            replace=False
                        )
                        df_images.loc[pos_place_type, 'place_type'] = 1

                        



        ##################
        # 3. assign to each bird type and place combination an image file
        ##################


        # which places to add
        target_places = [
            ['bamboo_forest', 'forest-broadleaf'],  # Land backgrounds
            ['ocean', 'lake-natural']]              # Water backgrounds


        # check; training, validation and test distribution 
        for split, split_label in [(0, 'train'), (1, 'val'), (2, 'test')]:
            print(f"{split_label}:")
            split_df = df_images.loc[df_images['split'] == split, :]
            print(f"waterbirds are {np.mean(split_df['y']):.3f} of the examples")
            print(f"y = 0, c = 0: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 0):.3f}, n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 0))}")
            print(f"y = 0, c = 1: {np.mean(split_df.loc[split_df['y'] == 0, 'place'] == 1):.3f}, n = {np.sum((split_df['y'] == 0) & (split_df['place'] == 1))}")
            print(f"y = 1, c = 0: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 0):.3f}, n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 0))}")
            print(f"y = 1, c = 1: {np.mean(split_df.loc[split_df['y'] == 1, 'place'] == 1):.3f}, n = {np.sum((split_df['y'] == 1) & (split_df['place'] == 1))}")

            if set_confounder_water:
                print(f"y = 0, c = 0, z = 0: {np.mean(split_df.loc[(split_df['y'] == 0) & (split_df['place'] == 0), 'place_type'] == 0):.3f})")
                print(f"y = 0, c = 0, z = 1: {np.mean(split_df.loc[(split_df['y'] == 0) & (split_df['place'] == 0), 'place_type'] == 1):.3f})")
                print(f"y = 0, c = 1, z = 0: {np.mean(split_df.loc[(split_df['y'] == 0) & (split_df['place'] == 1), 'place_type'] == 0):.3f})")
                print(f"y = 0, c = 1, z = 1: {np.mean(split_df.loc[(split_df['y'] == 0) & (split_df['place'] == 1), 'place_type'] == 1):.3f})")
                print(f"y = 1, c = 0, z = 0: {np.mean(split_df.loc[(split_df['y'] == 1) & (split_df['place'] == 0), 'place_type'] == 0):.3f})")
                print(f"y = 1, c = 0, z = 1: {np.mean(split_df.loc[(split_df['y'] == 1) & (split_df['place'] == 0), 'place_type'] == 1):.3f})")
                print(f"y = 1, c = 1, z = 0: {np.mean(split_df.loc[(split_df['y'] == 1) & (split_df['place'] == 1), 'place_type'] == 0):.3f})")
                print(f"y = 1, c = 1, z = 1: {np.mean(split_df.loc[(split_df['y'] == 1) & (split_df['place'] == 1), 'place_type'] == 1):.3f})")


        # folder belonging to each place
        place_ids_df = pd.read_csv(
            place_dir+ '/categories_places365.txt',
            sep=" ",
            header=None,
            names=['place_name', 'place_id'],
            index_col='place_id')

        # list with target place ids
        target_place_ids = []

        if not set_confounder_water:
            # go over [bamboo_forest, forest_broadleaf] and [ocean, lake_natural]
            for idx, target_places_category in enumerate(target_places):
                place_filenames = []
                print(f'category {idx} {target_places_category}')
                
                # go over each place type
                for target_place in target_places_category:
                    target_place_ids.append(target_place)
                    
                    print(f'category {idx} {target_place} has id {target_place_ids[idx]}')
                    
                    # get filename of places 
                    place_filenames += [
                        f'{target_place}/{filename}' for filename in os.listdir(
                            os.path.join(place_dir,target_place))
                        if filename.endswith('.jpg')]

                    print(f'number of files in {target_place}: {len(place_filenames)}')
                
                # shuffle the place filenames 
                random.shuffle(place_filenames)

                # Assign each filename to an image
                indices = (df_images.loc[:, 'place'] == idx)
                print('first 5 elements of indices', indices[:5])
                assert len(place_filenames) >= np.sum(indices),\
                    f"Not enough places ({len(place_filenames)}) to fit the dataset ({np.sum(df_images.loc[:, 'place'] == idx)})"
                df_images.loc[indices, 'place_filename'] = place_filenames[:np.sum(indices)]

                print(f"Assigned {np.sum(indices)} images to category {idx} {target_places_category}")


        else:

            for label, target_places_category in enumerate(target_places):
                print(f'category {label} {target_places_category}')
                
                # go over each place type
                i = 0 
                for target_place in target_places_category:
                    target_place_ids.append(target_place)
                    
                    print(f'category {label} {target_place} has id {target_place_ids[label]}')
                    
                    # get filename of places 
                    place_filenames = [
                        f'{target_place}/{filename}' for filename in os.listdir(
                            os.path.join(place_dir,target_place))
                        if filename.endswith('.jpg')]

                    print(f'number of files in {target_place}: {len(place_filenames)}')
                
                    # shuffle the place filenames 
                    random.shuffle(place_filenames)

                    # Assign each filename to an image
                    indices = (df_images.loc[:, 'place'] == label) & (df_images.loc[:, 'place_type'] == i)
                    print('number of images for place = {}, type = {} : {}'.format(i, label, np.sum(indices)))
                    assert len(place_filenames) >= np.sum(indices),\
                        f"Not enough places ({len(place_filenames)}) to fit the dataset ({np.sum(df_images.loc[:, 'place'] == label)})"
                    df_images.loc[indices, 'place_filename'] = place_filenames[:np.sum(indices)]

                    print(f"Assigned {np.sum(indices)} images to category {label} {target_places_category}")
                    i += 1

        print(df_images.head())
        print(df_images['place_filename'].head())

        df_images_train = df_images.loc[df_images['split'] == 0, :]
        df_images_train_check = df_images_train.groupby(['y', 'place', 'place_type']).size()
        print(df_images_train_check.shape)
        print(df_images_train_check)

        place_type_train_place_1 = df_images_train[df_images_train['place'] == 1]['place_filename'].str.split('/').str[0]
        print(place_type_train_place_1.value_counts())




        ###################
        # 4. Combine filenames 
        ###################

        image_path ='raw_input/birds'
        segmentation_path = 'raw_input/segmentations'

        for i in tqdm(df_images.index):
            
            # image of the bird
            img_path = os.path.join(image_path, df_images.loc[i, 'img_filename'])
            
            # get the image of the segmenation
            seg_path = os.path.join(segmentation_path, df_images.loc[i, 'img_filename'].replace('.jpg', '.png'))
            
            # get the place path 
            place_path = os.path.join(place_dir, df_images.loc[i, 'place_filename'])
            place = Image.open(place_path).convert('RGB')
            
            # set images to numpy
            img_np = np.asarray(Image.open(img_path).convert('RGB'))
            seg_np = np.asarray(Image.open(seg_path).convert('RGB'))/255
                
            # create image of bird with black background
            img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
            
            # create image of bird with place background
            combined_img = combine_and_mask(place, seg_np, img_black)
            
            # create image of bird only
            bird_img = combine_and_mask(Image.fromarray(np.ones_like(place) * 150), seg_np, img_black)
            
            # select image of place
            seg_np *= 0.
            img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
            place_img = combine_and_mask(place, seg_np * 0, img_black)

            # call the path for each type of data
            combined_path = os.path.join(output_dir, "combined", df_images.loc[i, 'img_filename'])
            bird_path = os.path.join(output_dir, "birds", df_images.loc[i, 'img_filename'])
            place_path = os.path.join(output_dir, "places", df_images.loc[i, 'img_filename'])
            
            # make directory
            os.makedirs('/'.join(combined_path.split('/')[:-1]), exist_ok=True)
            os.makedirs('/'.join(bird_path.split('/')[:-1]), exist_ok=True)
            os.makedirs('/'.join(place_path.split('/')[:-1]), exist_ok=True)
            
            
            # save images to folder 
            combined_img.save(combined_path)
            bird_img.save(bird_path)
            place_img.save(place_path)

        # write csv of metadata
        df_images.to_csv(os.path.join(output_dir, filename_metadata))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--confounder_strength_values', help ='list of confounder strength values')
    parser.add_argument('--set_confounder_water', help ='set confounder water')
    parser.add_argument('--confounder_strength_water',    help ='confounder strength water')


    args = parser.parse_args()
    dict_arguments = vars(args)
    confounder_strength_values = Convert(dict_arguments['confounder_strength_values'], float)
    set_confounder_water = str_to_bool(dict_arguments['set_confounder_water'])
    confounder_strength_water = float(dict_arguments['confounder_strength_water'])
    main(confounder_strength_values,set_confounder_water, confounder_strength_water )