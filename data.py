

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import torchvision 
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import math
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import numpy as np
import os
import random

from JSE.helpers import *
from JSE.models import *



class Dataset_obj():
    """
    Class for all datasets used

    """

    def get_k_eigenvectors(self, X, k, make_plot = False):
        """
        Calculates the k eigenvectors of X
        """

        # get the k eigenvectors
        U, S, V = torch.pca_lowrank(X, center= True, q = k)

        # if make plot, plot the variance explained
        if make_plot:
            U, S, V = torch.pca_lowrank(X, center= True, q = X.shape[1])
            eigenvalues = S**2/(X.shape[0] - 1)
            variance_explained = torch.cumsum(eigenvalues, dim = 0)/torch.sum(eigenvalues)
            plt.plot(list(range(1 , k  + 1)), variance_explained[:k])
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance')
            plt.rcParams['savefig.dpi'] = 500
            plt.show()
            sys.exit()

        return V[:, :k]

    @staticmethod
    def get_groups( y_c, y_m):
        """
        Get the groups for the dataset.
        """

        # get the groups
        groups = []
        for i in range(len(y_c)):
            group = return_group(y_c[i], y_m[i])
            groups.append(group)

        return groups

    def transform_data_to_k_components(self, k, include_test = False, reset_V_k = False):
        """
        Turn the data into k components
        """

        # get the k eigenvectors
        if reset_V_k:
           self.V_k_train = self.get_k_eigenvectors(self.X_train, k)

        # transform the data to the k components
        self.X_train = torch.matmul(self.X_train, self.V_k_train)
        self.X_val = torch.matmul(self.X_val, self.V_k_train)

        # transform the test data to the k components
        if include_test:
            self.X_test = torch.matmul(self.X_test, self.V_k_train)
   

    def demean_X(self, include_test = False, reset_mean = False):
        """
        Demean the data
        """

        # estimate the mean based on the training data
        if reset_mean:
            self.X_train_mean = self.X_train.mean(dim = 0)  
            self.X_val_mean = self.X_val.mean(dim = 0)      

        # demean the data based on the mean estimated from the training data 
        self.X_train = self.X_train - self.X_train_mean
        self.X_val = self.X_val - self.X_val_mean

        # demean the test data based on the mean estimated from the training data
        if include_test:
            self.X_test = self.X_test - self.X_train_mean #self.X_train_mean #TODO: check if valid
    
    @staticmethod 
    def get_n_per_class( y, classes = [0, 1] ):
        """
        Get the number of samples per class
        """

        # from y_c, get the count per class
        count_class = torch.bincount(y)
        n_per_class = dict(zip(classes, count_class))

        return n_per_class
    
    def get_group_size(self, subset_size, spurious_ratio ):
        """
        Get the group size for a subset of size subset_size, with spurious ratio spurious_ratio
        """
        
        # get the number of samples
        n_1 = int(subset_size * spurious_ratio )
        n_0 = int(subset_size * (1 - spurious_ratio) )

        return n_1, n_0
    
    def get_group_sizes_split(self, subset_size_1, subset_size_0, spurious_ratio):
        """
        Get the group sizes for a subset of size subset_size, for both classes ,with spurious ratio spurious_ratio
        """
        # get the number of samples
        n_1_0, n_1_1 = self.get_group_size(subset_size_1, spurious_ratio)
        n_0_0, n_0_1 = self.get_group_size(subset_size_0, spurious_ratio)


        return n_1_0, n_1_1, n_0_0, n_0_1

    def get_sample_for_group(self, y_m, y_c, size, y_m_group, y_c_group):
        """
        Get a sample of size size for a group y_m = y_m_group, y_c = y_c_group
        """

        # get the indices where y_m == y_m_group and y_c == y_c_group
        i_group = torch.where((y_m == y_m_group) & (y_c == y_c_group))[0]
        print(' for group y_m = {}, y_c = {} there are {} samples'.format(y_m_group, y_c_group, i_group.shape[0]))

        # get the sample of size, 
        if i_group.shape[0] < size:
            print('not enough samples for group y_m = {}, y_c = {}'.format(y_m_group, y_c_group))
            print('there are {} samples, but {} samples are needed'.format(i_group.shape[0], size))
            sys.exit()
        i_sample = torch.randint(0, i_group.shape[0], (size,))
        chosen_ids = i_group[i_sample]
       
        return chosen_ids
    
    def set_ids_for_split(self, n_1_1, n_1_0, n_0_0, n_0_1, y_m_split, y_c_split):
        """
        Get the ids for each split of the data
        """

        ids = []
        
        # loop over the groups
        for y_m_value in [0, 1]:
            for y_c_value in [0, 1]:

                # set the size
                if y_m_value == 1:
                    if y_c_value == 1:
                        size = n_1_1
                    else:
                        size = n_1_0
                else:
                    if y_c_value == 1:
                        size = n_0_1
                    else:
                        size = n_0_0

                
                ids_group = self.get_sample_for_group(y_m_split, y_c_split, size, y_m_value, y_c_value)
                ids.append(ids_group)
                print('size for group {} {} is {}'.format(y_m_value, y_c_value, size))

        return torch.cat(ids)
        
    def set_sample_split(self, y_m_split, y_c_split, n_split, spurious_ratio, p_m,  seed):
        """
        Set the sample split for the data
        """

        # set the seed
        set_seed(seed)

        # get the number of samples for y_m = 1, y_m = 0
        y_m_1 = int(n_split * p_m)
        y_m_0 = n_split - y_m_1
        print('Number of samples for y_m = 1: {}'.format(y_m_1))
        print('Number of samples for y_m = 0: {}'.format(y_m_0))

        # get the group sizes
        n_1_1, n_1_0, n_0_0, n_0_1 = self.get_group_sizes_split(y_m_1, y_m_0, spurious_ratio)
        print('Number of samples for y_m = 1, y_c = 1: {}'.format(n_1_1))
        print('Number of samples for y_m = 1, y_c = 0: {}'.format(n_1_0))
        print('Number of samples for y_m = 0, y_c = 1: {}'.format(n_0_1))
        print('Number of samples for y_m = 0, y_c = 0: {}'.format(n_0_0))

        # get the sample ids
        sample_ids = self.set_ids_for_split(n_1_1, n_1_0, n_0_0, n_0_1, y_m_split, y_c_split)

        return sample_ids

    @staticmethod
    def give_balanced_weights(groups, n_total, n_per_group, normalized=True):
        """
        Give balanced weights to the samples based on the groups
        """

        # store the weights in list
        list_weights = [None]* n_total
        n_groups = len(n_per_group.keys())
        i= 0
        total_weights = 0

        # loop over the groups
        encountered_weights = {}
        for group_index in groups:

            # get the weight
            n_group = n_per_group[group_index]
            weight = n_total / (n_groups * n_group)
            
            # if normalized, normalize the weights
            if normalized:
                encountered_groups = list(encountered_weights.keys())
                if group_index not in encountered_groups:
                    encountered_weights[group_index] = weight
                    total_weights += weight
                    print('Weight added for group with perc. samples {}: {}'.format(n_group/n_total, weight))
            
            list_weights[i] = weight
            i += 1
        
        # if normalized, normalize the weights
        if normalized:
            weights = torch.tensor(list_weights)/total_weights
        else:
            weights = torch.tensor(list_weights)
        return weights

    def reset_X_c_orth(self, P_c_orth, batch_size, workers = 0, shuffle=True, include_test=False, include_weights = False, train_weights = None, val_weights = None, only_main= True, reset_X_objects=True):
        """
        Reset the X based on the projection P_c_orth
        """

        # create X_c_orth
        X_c_orth_train = torch.matmul(self.X_train, P_c_orth)
        X_c_orth_val = torch.matmul(self.X_val, P_c_orth)

        # reset the X
        self.reset_X(X_c_orth_train, X_c_orth_val, batch_size, workers, shuffle, only_main = only_main, reset_X_objects=reset_X_objects, include_weights = include_weights, train_weights = train_weights, val_weights = val_weights)

        # if include test, apply here as well the projection
        if include_test:
            self.X_test = torch.matmul(self.X_test, P_c_orth)


    def reset_X(self, new_X_train, new_X_val, batch_size, workers=0, shuffle=True, only_main = False, reset_X_objects = True, include_weights = False, train_weights = None, val_weights = None, concept_first=True):
        """
        Reset the X, given new X train, val
        """

        # define X train, val
        if reset_X_objects:
            self.X_train =new_X_train
            self.X_val = new_X_val

        # if not only_main: reset the loaders involving concept labels
        if not only_main:
            self.concept_main_loader = self.create_loaders(batch_size, workers, shuffle=shuffle, with_concept=True, include_weights= include_weights, train_weights = train_weights, val_weights = val_weights, concept_first=concept_first)
            self.concept_loader = self.create_loaders(batch_size, workers, shuffle=shuffle, with_concept=False, which_dependent = 'concept', include_weights= include_weights, train_weights = train_weights, val_weights = val_weights)
        
        # reset the loaders involving main labels
        self.main_loader = self.create_loaders(batch_size, workers, shuffle=shuffle, with_concept=False, which_dependent='main', include_weights= include_weights, train_weights = train_weights, val_weights = val_weights)

    def get_class_weights_train_val(self, y_train, y_val):

        """
        Get the class weights for the training and validation set, for an y \in {0, 1}
        """

        # get number of images per class
        n_per_class_train = self.get_n_per_class( y_train)
        n_per_class_val = self.get_n_per_class( y_val)
        print('n_per_class_train', n_per_class_train)
        print('n_per_class_val', n_per_class_val)

        # get class weights
        self.class_weights_train = self.give_balanced_weights( y_train.tolist(), len(y_train), n_per_class_train)
        self.class_weights_val = self.give_balanced_weights( y_val.tolist(), len(y_val), n_per_class_val)
        
        return self.class_weights_train, self.class_weights_val
    
    def get_group_weights(self, normalized=True):
        """
        Get the group weights for the training and validation set, for an y_c, y_m \in {0, 1}
        """

        # get the groups
        groups_train = self.get_groups(self.y_c_train, self.y_m_train)
        groups_val = self.get_groups(self.y_c_val, self.y_m_val)

        n_per_group_train = self.get_n_per_class( torch.Tensor(groups_train).int(), classes= [0, 1,2,3])
        n_per_group_val = self.get_n_per_class( torch.Tensor(groups_val).int(), classes= [0, 1,2,3])

        # get the weights
        print('n_per_group_train', n_per_group_train)
        print('n_per_group_val', n_per_group_val)

        self.group_weights_train = self.give_balanced_weights( groups_train, len(groups_train), n_per_group_train, normalized=normalized)
        self.group_weights_val = self.give_balanced_weights( groups_val, len(groups_val), n_per_group_val, normalized=normalized)
        
        # return the weights
        return self.group_weights_train, self.group_weights_val

    def set_train_val_split(self, seed, train_split, df_meta = None):
        """
        Set train, val split. 
        """

        # set the seed
        set_seed(seed)

        # get train, val split
        i_shuffled = torch.randperm(self.n)
        self.i_train = i_shuffled[:int(train_split*self.n)]
        self.i_val = i_shuffled[int(train_split*self.n):]

        # define X train, val
        self.X_train = self.X[self.i_train, :]
        self.X_val = self.X[self.i_val, :]

        # define Y train, val
        self.y_c_train = self.y_c[self.i_train]
        self.y_c_val = self.y_c[self.i_val]
        
        # define Y train, val
        self.y_m_train = self.y_m[self.i_train]
        self.y_m_val = self.y_m[self.i_val]


    

    def create_loaders(self, batch_size, workers, shuffle=True, with_concept=True, which_dependent='main', include_weights=False, train_weights = None, val_weights = None, concept_first=True):
        """ 
        Create train and validation loaders
        """

        # if with concepts, add concept labels to the data
        if with_concept:
           if concept_first:
                train = TensorDataset(self.X_train, self.y_c_train, self.y_m_train)
                val = TensorDataset(self.X_val, self.y_c_val, self.y_m_val)
           else:
                train = TensorDataset(self.X_train, self.y_m_train, self.y_c_train)
                val = TensorDataset(self.X_val, self.y_m_val, self.y_c_val)
        # only one set of labels
        else:
            if which_dependent == 'main':
                train = TensorDataset(self.X_train, self.y_m_train)
                val = TensorDataset(self.X_val, self.y_m_val)
            elif which_dependent == 'concept':
                train = TensorDataset(self.X_train, self.y_c_train)
                val = TensorDataset(self.X_val, self.y_c_val)
        
        # if include weights,add weights to the sampler
        if include_weights:
            train_sampler = torch.utils.data.WeightedRandomSampler(train_weights, num_samples = len(train_weights))
            val_sampler = torch.utils.data.WeightedRandomSampler(val_weights, num_samples = len(val_weights))
            shuffle = False
        else:
            train_sampler = val_sampler = None

        # create the loaders
        self.train_loader = DataLoader(train, 
                                        batch_size=batch_size,
                                         shuffle=shuffle, 
                                         sampler =train_sampler,
                                         num_workers=workers)
        self.val_loader = DataLoader(val, 
                                        batch_size=batch_size, 
                                        shuffle=shuffle, 
                                        sampler = val_sampler,
                                        num_workers=workers)
        self.dict_loaders = {'train': self.train_loader, 'val': self.val_loader}

        return self.dict_loaders
    

    def create_loaders_for_BERT(self, batch_size, workers, sample=False, adversarial = False):
        """
        Create loaders for BERT, since the data is stored in a different way
        """

        # if sample, refer to different objects
        if sample:
            # if adversarial, add concept labels
            if adversarial:
                train = TensorDataset(self.input_ids_train_sample,  self.segment_ids_train_sample, self.input_masks_train_sample, self.y_m_train_sample, self.y_c_train_sample)
                val = TensorDataset(self.input_ids_val_sample, self.segment_ids_val_sample, self.input_masks_val_sample, self.y_m_val_sample, self.y_c_val_sample)
            else:
                train = TensorDataset(self.input_ids_train_sample,  self.segment_ids_train_sample, self.input_masks_train_sample, self.y_m_train_sample)
                val = TensorDataset(self.input_ids_val_sample, self.segment_ids_val_sample, self.input_masks_val_sample, self.y_m_val_sample)
        # if not sample, refer to the original objects
        else:

            train = TensorDataset(self.input_ids_train,  self.segment_ids_train,self.input_masks_train, self.y_m_train)
            val = TensorDataset(self.input_ids_val,self.segment_ids_val, self.input_masks_val,  self.y_m_val)

        # create the samplers
        train_sampler = RandomSampler(train)
        # sequential sampler for validation, since faster
        val_sampler = SequentialSampler(val)

        # create the loaders
        self.train_loader = DataLoader(train, sampler=train_sampler, batch_size=batch_size, num_workers= workers)
        self.val_loader = DataLoader(val, sampler=val_sampler, batch_size=batch_size, num_workers=workers) 
        self.dict_loaders = {'train': self.train_loader, 'val': self.val_loader}

        return self.dict_loaders
    


    def sample_balanced_test_set(self, seed):
        """
        Sample a balanced test set
        """
        
        # get the indices of the test set where y_m = 0, y_m = 1
        i_y_m_0 = torch.where(self.y_m_test == 0)
        i_y_m_1 = torch.where(self.y_m_test == 1)

        # sample a balanced set where p(y_m = 1 ) = 0.5
        set_seed(seed)

        # how many samples to take from each group
        n_small_set = min(len(i_y_m_0[0]), len(i_y_m_1[0]))

        # sample the indices
        random_indices_0 = torch.randint(0, len(i_y_m_0[0]), (n_small_set,))
        random_indices_1 = torch.randint(0, len(i_y_m_1[0]), (n_small_set,))
        i_y_m_0_balanced = i_y_m_0[0][random_indices_0]
        i_y_m_1_balanced = i_y_m_1[0][random_indices_1]
        i_y_m_balanced = torch.cat((i_y_m_0_balanced, i_y_m_1_balanced))
        
        # get the balanced test set
        y_m_test_balanced = self.y_m_test[i_y_m_balanced]
        y_c_test_balanced = self.y_c_test[i_y_m_balanced]
        X_test_balanced = self.X_test[i_y_m_balanced, :]

        return X_test_balanced, y_m_test_balanced, y_c_test_balanced


class multiNLI_Dataset(Dataset_obj):
    """
    Class for the multiNLI dataset
    """
    def __init__(self):
        super().__init__()

    def load_processed_dataset(self, filename, device):
        """
        Load the processed dataset
        """

        # load the data
        with open(filename, 'rb') as f:
            dict_data = pickle.load(f)

        # add the training data
        self.X_train = dict_data['train']['embeddings'].to(device)
        self.y_m_train = dict_data['train']['y_m'].to(device).int()
        self.y_c_train = dict_data['train']['y_c'].to(device).int()

        # add the validation data
        self.X_val = dict_data['val']['embeddings'].to(device)
        self.y_m_val = dict_data['val']['y_m'].to(device).int()
        self.y_c_val = dict_data['val']['y_c'].to(device).int()

        # add the test data
        self.X_test = dict_data['test']['embeddings'].to(device)
        self.y_m_test = dict_data['test']['y_m'].to(device).int()
        self.y_c_test = dict_data['test']['y_c'].to(device).int()

        # combine the train, validation data
        self.X = torch.cat([self.X_train, self.X_val], dim=0)
        self.y_m = torch.cat([self.y_m_train, self.y_m_val], dim=0)
        self.y_c = torch.cat([self.y_c_train, self.y_c_val], dim=0)
        self.i_train = torch.arange(0, self.X_train.shape[0])
        self.i_val = torch.arange(self.X_train.shape[0], self.X.shape[0])

        # Sense check: print the number of samples per class
        print('n samples where y_m = 1 and y_c = 1: ', (self.y_m_train[self.y_c_train == 1] == 1).sum())
        print('n samples where y_m = 1 and y_c = 0: ', (self.y_m_train[self.y_c_train == 0] == 1).sum())
        print('n samples where y_m = 0 and y_c = 1: ', (self.y_m_train[self.y_c_train == 1] == 0).sum())
        print('n samples where y_m = 0 and y_c = 0: ', (self.y_m_train[self.y_c_train == 0] == 0).sum())
        print('number of embeddings: ', self.X_train.shape[0])
        print('number of y_m: ', self.y_m_train.shape[0])
        print('number of y_c: ', self.y_c_train.shape[0])

    def set_sample_subset(self, n_train, n_val, n_test,  spurious_ratio,p_m,  seed):
        """
        Set a sample subset of the data, specifically for multiNLI
        """

        set_seed(seed)

        #y_m_split, y_c_split, n_split, spurious_ratio, p_m,  seed
        self.sample_train_ids = self.set_sample_split(self.y_m_train, self.y_c_train, n_train, spurious_ratio, p_m,  seed)
        self.sample_val_ids = self.set_sample_split(self.y_m_val, self.y_c_val, n_val, spurious_ratio, p_m,  seed)
        self.sample_test_ids = self.set_sample_split(self.y_m_test, self.y_c_test, n_test, 0.5, p_m,  seed)

        # define the train, val input ids
        self.input_ids_train_sample = self.input_ids_train[self.sample_train_ids, :]
        self.input_ids_val_sample = self.input_ids_train[self.sample_val_ids, :]
        self.input_ids_test_sample = self.input_ids_test[self.sample_test_ids, :]

        # define the train, val segment ids
        self.segment_ids_train_sample = self.segment_ids_train[self.sample_train_ids, :]
        self.segment_ids_val_sample = self.segment_ids_train[self.sample_val_ids, :]
        self.segment_ids_test_sample = self.segment_ids_test[self.sample_test_ids, :]

        # define the train, val input masks
        self.input_masks_train_sample = self.input_masks_train[self.sample_train_ids, :]
        self.input_masks_val_sample = self.input_masks_train[self.sample_val_ids, :]
        self.input_masks_test_sample = self.input_masks_test[self.sample_test_ids, :]

        # define the train, val labels
        self.y_m_train_sample = self.y_m_train[self.sample_train_ids]
        self.y_m_val_sample = self.y_m_train[self.sample_val_ids]
        self.y_m_test_sample = self.y_m_test[self.sample_test_ids]

        # define the train, val labels
        self.y_c_train_sample = self.y_c_train[self.sample_train_ids]
        self.y_c_val_sample = self.y_c_train[self.sample_val_ids]
        self.y_c_test_sample = self.y_c_test[self.sample_test_ids]

        # Sense check: print the number of samples per class
        print('Division of y_m/y_c in loaded data')
        print('n of samples where y_m = 1: {}'.format(torch.sum(self.y_m_train_sample)))
        print('n of samples where y_m = 0: {}'.format(torch.sum(self.y_m_train_sample==0)))
        print('n of samples where y_m = 1, y_c = 1: {}'.format(torch.sum((self.y_m_train_sample==1) & (self.y_c_train_sample==1))))
        print('n of samples where y_m = 1, y_c = 0: {}'.format(torch.sum((self.y_m_train_sample==1) & (self.y_c_train_sample==0))))
        print('n of samples where y_m = 0, y_c = 1: {}'.format(torch.sum((self.y_m_train_sample==0) & (self.y_c_train_sample==1))))
        print('n of samples where y_m = 0, y_c = 0: {}'.format(torch.sum((self.y_m_train_sample==0) & (self.y_c_train_sample==0))))

        

    
    def load_main_dataset(self,directory_features, directory_metadata, turn_binary=True, punctuation=True):
        """
        Load the original main dataset
        """

        # the version without punctuation
        if not punctuation:
            self.metadata_df = pd.read_csv(directory_metadata +  "/metadata_multiNLI.csv")

            # Load features
            self.features_array = []
            for feature_file in [
                'cached_train_bert-base-uncased_128_mnli',  
                'cached_dev_bert-base-uncased_128_mnli',
                'cached_dev_bert-base-uncased_128_mnli-mm'
                ]:
                print(os.path.join(
                        directory_features,
                        feature_file) )
                features = torch.load(
                    os.path.join(
                        directory_features,
                        feature_file))

                self.features_array += features
            
            # get the input ids, input masks, segment ids, label ids
            self.all_input_ids = torch.tensor([f.input_ids for f in self.features_array], dtype=torch.long)
            self.all_input_masks = torch.tensor([f.input_mask for f in self.features_array], dtype=torch.long)
            self.all_segment_ids = torch.tensor([f.segment_ids for f in self.features_array], dtype=torch.long)
            self.all_label_ids = torch.tensor([f.label_id for f in self.features_array], dtype=torch.long)
            self.y_c = torch.Tensor(self.metadata_df['sentence2_has_negation'].values)

        # the version with punctuation
        else:
            self.metadata_df = pd.read_csv(directory_metadata +  "/metadata_multiNLI_punctuated.csv")

            with open(directory_features + '/multiNLI_punctuated.pkl', 'rb') as f:
                dict_data = pickle.load(f)

            self.all_input_ids = dict_data['input_ids']
            self.all_input_masks = dict_data['attention_masks']
            self.all_segment_ids = dict_data['token_type_ids']
            self.y_c = dict_data['punctuation']

        # get the labels
        self.y_m = torch.Tensor(self.metadata_df['gold_label'].values)

        # if not turn_binary, turn to a 3-dimensional tensor, with [1, 0, 0] if 0, [0, 1, 0] if 1, etc. 
        if not turn_binary:
            self.y_m = torch.nn.functional.one_hot(self.y_m.to(torch.int64), num_classes=3).to(torch.float32)
        else:
            # return 0 if 0, 1 else 
            self.y_m  = (self.y_m == 0).squeeze(-1)
            
        # get the train_ids from the metadata
        self.train_ids = torch.Tensor(self.metadata_df[self.metadata_df['split'] == 0].index.values).long()
        self.val_ids = torch.Tensor(self.metadata_df[self.metadata_df['split'] == 1].index.values).long()
        self.test_ids = torch.Tensor(self.metadata_df[self.metadata_df['split'] == 2].index.values).long()

        # get the train, val, test input ids
        self.input_ids_train = self.all_input_ids[self.train_ids, :]
        self.input_ids_val = self.all_input_ids[self.val_ids, :]
        self.input_ids_test = self.all_input_ids[self.test_ids, :]

        # get the train, val, test input masks
        self.input_masks_train = self.all_input_masks[self.train_ids, :]
        self.input_masks_val = self.all_input_masks[self.val_ids, :]
        self.input_masks_test = self.all_input_masks[self.test_ids, :]

        # get the train, val, test segment ids
        self.segment_ids_train = self.all_segment_ids[self.train_ids, :]
        self.segment_ids_val = self.all_segment_ids[self.val_ids, :]
        self.segment_ids_test = self.all_segment_ids[self.test_ids, :]

        # get the train, val, test labels for main
        if not turn_binary:
            self.y_m_train = self.y_m[self.train_ids, ]
            self.y_m_val = self.y_m[self.val_ids, ]
            self.y_m_test = self.y_m[self.test_ids, ]
        else:
            self.y_m_train = self.y_m[self.train_ids]
            self.y_m_val = self.y_m[self.val_ids]
            self.y_m_test = self.y_m[self.test_ids]

        # get the train, val, test labels for concept
        self.y_c_train = self.y_c[self.train_ids]
        self.y_c_val = self.y_c[self.val_ids]
        self.y_c_test = self.y_c[self.test_ids]

    def get_group_sizes(self, y_m, y_c):
        """
        Get the size per group, based on y_m, y_c
        """

        # get number of samples per class
        n_1_m = torch.sum(y_m == 1)
        n_1_c = torch.sum(y_c == 1)
        print('p(y_m = 1): ', n_1_m / len(y_m))
        print('p(y_c = 1): ', n_1_c / len(y_c))

        # get number of samples per group
        n_1_0 = torch.sum((y_m == 1) & (y_c == 0))
        n_1_1 = torch.sum((y_m == 1) & (y_c == 1))
        n_0_0 = torch.sum((y_m == 0) & (y_c == 0))
        n_0_1 = torch.sum((y_m == 0) & (y_c == 1))

        print('Group sizes:')
        print('Sample size for y_m = 1, y_c = 1: ', n_1_1)
        print('Sample size for y_m = 1, y_c = 0: ', n_1_0)
        print('Sample size for y_m = 0, y_c = 1: ', n_0_1)
        print('Sample size for y_m = 0, y_c = 0: ', n_0_0)

 
    def return_train_val_test_data(self):
        return self.input_ids_train, self.input_ids_val, self.input_ids_test, self.input_masks_train, self.input_masks_val, self.input_masks_test, self.segment_ids_train, self.segment_ids_val, self.segment_ids_test, self.y_m_train, self.y_m_val, self.y_m_test, self.y_c_train, self.y_c_val, self.y_c_test







class Waterbird_Dataset(Dataset_obj):
    """
    Class for the waterbird dataset
    """
    def __init__(self):
        super().__init__()

    
    def transform_cub(self, target_resolution):
        """
        Transform the images to the target resolution, normalize to ImageNet standards
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

    def load_original_images(self, spurious_ratio, type_image):
        """
        Load the original images from the Waterbird dataset, for a given spurious ratio and type of image 
        """

        # select metadata
        spurious_ratio_str = str(int(spurious_ratio*100))
        folder_name = 'images/waterbirds_'+spurious_ratio_str
        
        # set the metadata filename
        meta_data_filename = folder_name + '/metadata_waterbird_'+spurious_ratio_str+ '_' +spurious_ratio_str + '_50'
        meta_data_filename += '.csv'

        # get the metadata file
        df_images = pd.read_csv(meta_data_filename)

        # gather info per train, validation and test set
        df_images_train = df_images[df_images['split']==0]
        df_images_val = df_images[df_images['split']==1]
        df_images_test = df_images[df_images['split']==2]

        # change to (224, 224) resolution
        resolution_waterbird = (224, 224)

        # define transformation function
        tranformation_waterbird = self.transform_cub(resolution_waterbird)

        # Check whether the specified path exists or not
        tensor_folder = folder_name +'/tensors'
        folder_exists = os.path.exists(tensor_folder)
        if not folder_exists:
                # Create a new directory because it does not exist
                os.makedirs(tensor_folder)


        waterbird_images_combined_train, waterbird_labels_combined_train = load_images(df_images_train, tranformation_waterbird,folder_name+'/', type_image )
        waterbird_images_combined_val, waterbird_labels_combined_val = load_images(df_images_val, tranformation_waterbird,folder_name+'/', type_image )
        waterbird_images_combined_test, waterbird_labels_combined_test = load_images(df_images_test, tranformation_waterbird,folder_name+'/', type_image )

        # save the tensors - training
        self.X_train = waterbird_images_combined_train
        self.y_m_train = waterbird_labels_combined_train
        self.y_c_train = torch.Tensor(df_images_train['place'].values)

        # save the tensors - validation
        self.X_val = waterbird_images_combined_val
        self.y_m_val = waterbird_labels_combined_val
        self.y_c_val = torch.Tensor(df_images_val['place'].values)
   
    def load_dataset(self, spurious_ratio, directory_labels, directory_embeddings, directory_labels_test, directory_embeddings_test, df_meta,df_meta_test, device, model_name='resnet50', image_type_train_val='combined', image_type_test='combined', corr_test='50', adversarial=False, seed=1, balance_main=False):
        """
        Load the dataset (of embeddings), for a given spurious ratio, model name, image type and correlation test
        """

        #Load the embeddings
        train_embedding_name = 'waterbird_images_{}_train_{}_embeddings_{}'.format(image_type_train_val,spurious_ratio, model_name)
        if adversarial:
            train_embedding_name += '_finetuned_adversarial' 
            train_embedding_name += '_seed_{}'.format(seed)


        train_embeddings = torch.load( directory_embeddings +'/' +train_embedding_name + '.pt')
        train_labels = torch.load(directory_labels +'/' + 'waterbird_labels_{}_train_{}.pt'.format(image_type_train_val,spurious_ratio))
           
        # validation set
        val_embedding_name = 'waterbird_images_{}_val_{}_embeddings_{}'.format(image_type_train_val,spurious_ratio, model_name)
        if adversarial:
            val_embedding_name += '_finetuned_adversarial'
            val_embedding_name += '_seed_{}'.format(seed)
        val_embeddings = torch.load(directory_embeddings +'/' + val_embedding_name + '.pt')
        val_labels = torch.load(directory_labels +'/' + 'waterbird_labels_{}_val_{}.pt'.format(image_type_train_val, spurious_ratio))

        # test set 
        test_embedding_name = 'waterbird_images_{}_test_{}_embeddings_{}'.format(image_type_test,corr_test, model_name)
        if adversarial:
            test_embedding_name += '_finetuned_adversarial'
            test_embedding_name += '_seed_{}'.format(seed)
        test_embeddings = torch.load(directory_embeddings_test + '/'+ test_embedding_name + '.pt')
        test_labels = torch.load(directory_labels_test + '/' + 'waterbird_labels_{}_test_{}.pt'.format(image_type_test, corr_test))

        # define X
        self.i_train = torch.tensor(list(range(train_embeddings.shape[0])))
        self.i_val = torch.tensor(list(range(train_embeddings.shape[0], train_embeddings.shape[0] + val_embeddings.shape[0])))
        self.X_train =train_embeddings
        self.X_val = val_embeddings
        self.X_test = test_embeddings

        # set to device
        self.X_train = self.X_train.to(device)
        self.X_val = self.X_val.to(device)
        self.X_test = self.X_test.to(device)


        # define y_m
        self.y_m = torch.cat((train_labels, val_labels), 0)
        self.y_m_train = train_labels
        self.y_m_val = val_labels
        self.y_m_test = test_labels


        # define y_c 
        # dataframe with meta data
        self.df_meta_train = df_meta[df_meta['split']==0]
        self.df_meta_val = df_meta[df_meta['split']==1]
        self.df_meta_test = df_meta_test[df_meta_test['split']==2]

       
        self.y_c_train = torch.tensor(self.df_meta_train['place'].values)
        self.y_c_val = torch.tensor(self.df_meta_val['place'].values)        
        self.y_c_test = torch.tensor(self.df_meta_test['place'].values)

        # define the train, val, test indeces
        if balance_main:
            
            # sample sizes of waterbirds
            n_of_waterbird_images_train = (self.y_m_train == 1).sum()
            n_of_waterbird_images_val = (self.y_m_val == 1).sum()
            n_of_waterbird_images_test = (self.y_m_test == 1).sum()

            # sample n_of_waterbird_images_train images where y_m == 0
            i_y_m_0_train = torch.where(self.y_m_train == 0)[0]
            i_y_m_0_val = torch.where(self.y_m_val == 0)[0]
            i_y_m_0_test = torch.where(self.y_m_test == 0)[0]
            i_y_m_1_train = torch.where(self.y_m_train == 1)[0]
            i_y_m_1_val = torch.where(self.y_m_val == 1)[0]
            i_y_m_1_test = torch.where(self.y_m_test == 1)[0]

            # sample n_of_waterbird_images_train images where y_m == 0
            random_indices_0_train = torch.randint(0, len(i_y_m_0_train), (n_of_waterbird_images_train,))
            random_indices_0_val = torch.randint(0, len(i_y_m_0_val), (n_of_waterbird_images_val,))
            random_indices_0_test = torch.randint(0, len(i_y_m_0_test), (n_of_waterbird_images_test,))
            i_y_m_0_balanced_train = i_y_m_0_train[random_indices_0_train]
            i_y_m_0_balanced_val = i_y_m_0_val[random_indices_0_val]
            i_y_m_0_balanced_test = i_y_m_0_test[random_indices_0_test]

            # sample n_of_waterbird_images_train images where y_m == 1
            random_indices_1_train = torch.randint(0, len(i_y_m_1_train), (n_of_waterbird_images_train,))
            random_indices_1_val = torch.randint(0, len(i_y_m_1_val), (n_of_waterbird_images_val,))
            random_indices_1_test = torch.randint(0, len(i_y_m_1_test), (n_of_waterbird_images_test,))
            i_y_m_1_balanced_train = i_y_m_1_train[random_indices_1_train]
            i_y_m_1_balanced_val = i_y_m_1_val[random_indices_1_val]
            i_y_m_1_balanced_test = i_y_m_1_test[random_indices_1_test]

            # get the balanced train, val, test set
            i_y_m_balanced_train = torch.cat((i_y_m_0_balanced_train, i_y_m_1_balanced_train))
            i_y_m_balanced_val = torch.cat((i_y_m_0_balanced_val, i_y_m_1_balanced_val))
            i_y_m_balanced_test = torch.cat((i_y_m_0_balanced_test, i_y_m_1_balanced_test))

            # define the train, val, test indeces
            self.i_train = i_y_m_balanced_train
            self.i_val = i_y_m_balanced_val
            self.i_test = i_y_m_balanced_test

            # define the train, val, test input ids
            self.X_train = self.X_train[self.i_train, :]
            self.X_val = self.X_val[self.i_val, :]
            self.X_test = self.X_test[self.i_test, :]

            # define the train, val, test labels
            self.y_m_train = self.y_m_train[self.i_train]
            self.y_m_val = self.y_m_val[self.i_val]
            self.y_m_test = self.y_m_test[self.i_test]

            # define the train, val, test labels
            self.y_c_train = self.y_c_train[self.i_train]
            self.y_c_val = self.y_c_val[self.i_val]
            self.y_c_test = self.y_c_test[self.i_test]


class Toy_Dataset(Dataset_obj):
  """
  Class for the toy dataset
  """
  def __init__(self):
    super().__init__()


  def draw_multivariate_normal(self, mu, sigma, n):
        """
        Draw from a multivariate normal distribution
        """
        mv = MultivariateNormal(mu, covariance_matrix=sigma)
        
        draw = mv.sample((n, ))
        
        return draw



  def draw_normal(self, mu, sigma, n):
        """
        Draw from a normal distribution
        """
        mv = Normal(mu, scale=sigma)
        
        draw = mv.sample((n, ))
        
        return draw

  def inv_logit_probability(self,X, beta, intercept):
        """
        Compute the inverse logit probability
        """

        z = torch.matmul(X, beta) + intercept

        return 1/(1 + torch.exp(-z))



  def define_v_unit(self, angle, gamma_m, d, second_main_feature_index, eps = 1e-8, adjustment = True, ):
    """
    Define the unit vector v_m, based on the angle, gamma_m, d, second_main_feature_index, eps, adjustment
    """

   
    # Convert the angle to radians
    angle_rad = math.radians(angle)

    # Calculate the x and y components of the new vector
    v_m_1 = math.cos(angle_rad)
    v_m_2 = math.sin(angle_rad)

    # define an vector of zeros, size d
    v_m = torch.zeros(d)

    # define the first and second main feature
    v_m[1] = v_m_1
    v_m[second_main_feature_index] = v_m_2

    # ensure that the vector is a unit vector
    v_m[v_m <= eps] = 0

    # adjust the vector to have the desired gamma_m
    if adjustment:
        if v_m_1 == 0 or v_m_2 == 0:
            v_m = v_m*gamma_m
        else:
            v_m[0] = gamma_m/( 1+ (v_m_1/v_m_2))
            v_m[1] = gamma_m/( 1+ (v_m_2/v_m_1))
       

   
    return v_m

  
  def plot_groups(self,ax_to_add, y_m, y_c, X, 
                colors= ['green', 'blue'], markers = ['.', 'P'], alpha=0.5, s=20, 
                groups = [1,2,3,4],  X_1_index = 0, X_2_index = 1, facecolor_type='fill', adendum_label='', legend=True, legend_loc='lower right' , X_1_line=4, X_2_line=4, X_1_plot = None, X_2_plot = None, edgecolor_type='fill'):
     """
     Function to plot the groups in a scatter plot
     """
     
     # get groups
     group_1 = (y_m==1) & (y_c==1)
     group_2 = (y_m==1) & (y_c==0)
     group_3 = (y_m==0) & (y_c==1)
     group_4 = (y_m==0) & (y_c==0)
     labels = [r'$y_{i, \mathrm{sp}} = 1, y_{i, \mathrm{mt}} = 1$', r'$y_{i, \mathrm{sp}} = 0, y_{i, \mathrm{mt}} = 1$', r'$y_{i, \mathrm{sp}} = 1, y_{i, \mathrm{mt}} = 0$', r'$y_{i, \mathrm{sp}} = 0, y_{i, \mathrm{mt}} = 0$']
     dict_groups = {1: group_1, 2: group_2, 3: group_3, 4: group_4}


     if X_1_plot is None:
        X_1_plot = X[:, X_1_index]
     if X_2_plot is None:
        X_2_plot = X[:, X_2_index]
     
     
     # loop over groups
     i = 0
     for group in groups:

         group_i = dict_groups[group]
         if group == 1 or group == 2:
             color = colors[0]
         else:
             color = colors[1]
         
         if group ==2 or group ==4:
             marker = markers[1]
         else:
             marker=markers[0]

         if facecolor_type == 'fill':
            facecolor = color
         else:
            facecolor = 'none'
         print('the fill color is {}'.format(facecolor))


         if edgecolor_type == 'fill':
            edgecolor = color
         else:
            edgecolor = edgecolor_type

         label = '{}'.format(labels[i]) +adendum_label
         ax_to_add.scatter(X_1_plot[group_i,], X_2_plot[group_i,], color=color, marker=marker, label=label,facecolors=facecolor, alpha=alpha,s=s, edgecolors=edgecolor)
         i += 1
        
    
     # plot lines
     ax_to_add.hlines(0, xmin= -X_1_line, xmax = X_1_line,color ='black', linestyles='-' , alpha=0.5)
     ax_to_add.vlines(0, ymin= -X_2_line, ymax = X_1_line, color='black' , linestyles='-', alpha=0.5)
     if legend:
        ax_to_add.legend(bbox_to_anchor=(1.0, .80), loc=legend_loc)
    
  def sample_data(self, n, d, rho_c_m, rho_c, rho_m,  gamma_m, gamma_c, intercept_main, intercept_concept, X_variance, angle, n_main_features = 1, n_concept_features = 1 ):
      """
      Sample data for the Toy dataset, based on the parameters
      """
      
      # set the number of samples
      self.n = n

      # set the mean for the multivariate normal
      mu = torch.zeros(d)
      
      # set the main-task vector  
      v_m = self.define_v_unit(angle, gamma_m, d, second_main_feature_index=0, eps = 1e-8, adjustment = True, )

      # set the concept vector
      v_c = torch.zeros(d)
      v_c[0] = 1 * gamma_c

      print('v_m/ v_c', v_m, v_c)

      # set the covariance matrix
      sigma = torch.eye(d) * X_variance

      # set correlation between the main and concept task feature
      sigma[0, 1] = rho_c_m
      sigma[1, 0] = rho_c_m
      
      # set correlation between the first column (concept feature) and the last n_concept_features columns
      if n_concept_features > 1:

        sigma[0, 2:(n_concept_features+1)] = rho_c
        sigma[2:(n_concept_features+1), 0] = rho_c

        sigma[1, 2:(n_concept_features+1)] = rho_c_m
        sigma[2:(n_concept_features+1), 1] = rho_c_m



      # set correlation between the second column (main feature) and the last n_main_features columns
      if n_main_features > 1:
          sigma[1, (n_concept_features+1):(n_concept_features+1+n_main_features)] = rho_m
          sigma[(n_concept_features+1):(n_concept_features+1+n_main_features), 1] = rho_m


     
      # define datapoints X
      X = self.draw_multivariate_normal(mu, sigma, n)

      # define the concept labels
      p_c = self.inv_logit_probability(X, v_c, intercept_concept)
      p_m = self.inv_logit_probability(X, v_m, intercept_main)
      y_c = torch.bernoulli(p_c).int()
      y_m = torch.bernoulli(p_m).int()
      
      # set the attributes
      self.X = X
      self.d = X.shape[1]
      self.y_c = y_c.squeeze(-1)
      self.y_m = y_m.squeeze(-1)
      self.sigma = sigma

      return X,  y_c.squeeze(-1), y_m.squeeze(-1), sigma

  




class celebA_Dataset(Dataset_obj):
  """
  Class for the celebA dataset
  """
  def __init__(self):
    super().__init__()

  def get_embeddings_in_chunks(self, step_size, df_metadata, transformation_func, folder, device, model):
    """
    Get the embeddings in chunks, in order to avoid memory issues
    """

    list_of_embedding_chunks = []
    list_of_label_chunks = []

    # takes steps of step_size and loads the images
    for step in range(0, len(df_metadata.index)+1, step_size):
        index_chunk = np.arange(step, step+step_size)
       
        if (step + step_size) > len(df_metadata.index):
            self.df_metadata_chunk = df_metadata.iloc[step:]
            print('Grabbing obs from .. onwards: ', step)
        else:
            self.df_metadata_chunk = df_metadata.iloc[index_chunk]
            print('Grabbing obs from .. to ..: ', step, step+step_size)

        

                
        # get all the images in the folder
        celebA_images_chunk, celebA_labels_chunk= load_images(self.df_metadata_chunk, transformation_func, folder, '')
        celebA_images_chunk = celebA_images_chunk.to(device)


         # put images in tensorDataset
        celebA_images= TensorDataset(celebA_images_chunk)
        

        # loaders for bird dataset
        batch_size=100
        workers=0
        loader = {
            'images' : DataLoader(celebA_images, 
                                                batch_size=batch_size, 
                                                shuffle=False, 
                                                num_workers=workers)
            }
        
         # embeddings for combined images
        celebA_embeddings = get_embedding_in_batches(model, loader['images'])

       
        # append the embeddings and labels to the list
        list_of_embedding_chunks.append(celebA_embeddings)
        list_of_label_chunks.append(celebA_labels_chunk)


    # flatten the lists
    embeddings = torch.cat(list_of_embedding_chunks)
    labels = torch.cat(list_of_label_chunks)

    print('embeddings shape: ', embeddings.shape)

    return embeddings, labels
  
  def set_dependent_var(self,  main_task_name, concept_name, folder):
    """
    Set the dependent variable, based on the main task name and concept name
    """

    df_att = pd.read_csv(folder + 'list_attr_celeba.txt', sep='\s+', skiprows=1)
    df_eval = pd.read_csv(folder + 'list_eval_partition.txt', sep='\s+', skiprows=0, header=None)
    df_eval.columns = ['image_id', 'split']

    df_att = df_att.reset_index(drop=False)
    print('df_att columns: ', df_att.columns)
    df_att.columns = ['image_id'] + list(df_att.columns[1:])


    # Add  a Female column by replacing -1 with 1 and vice versa
    df_att['Female'] = df_att['Male']*-1
    
    df_att['y'] = df_att[main_task_name].replace({-1:0})


    # merge the two dataframes on image_id
    df_metadata = pd.merge(df_att, df_eval, on='image_id')
    df_metadata['img_filename'] = df_metadata['image_id']
   
    # get the main-task and concept labels
    self.main_task_name = main_task_name
    self.concept_name = concept_name
    y_m = torch.tensor(df_metadata[main_task_name].replace({-1:0}).values)
    y_c = torch.tensor(df_metadata[concept_name].replace({-1:0}).values)
    print('total samples where y_m = 1: ', y_m.sum())
    print('total samples where y_c = 1: ', y_c.sum())
    print('total samples where y_m = 1 and y_c = 1: ', (y_m*y_c).sum())
    print('total samples where y_m = 1 and y_c = 0: ', (y_m*(1-y_c)).sum())
    print('total samples where y_m = 0 and y_c = 1: ', ((1-y_m)*y_c).sum())
    print('total samples where y_m = 0 and y_c = 0: ', ((1-y_m)*(1-y_c)).sum())

    # get indices for train, val and test from df_metadata
    self.df_metadata_train = df_metadata[df_metadata['split'] == 0]
    self.df_metadata_val = df_metadata[df_metadata['split'] == 1]
    self.df_metadata_test = df_metadata[df_metadata['split'] == 2]
    self.df_metadata = df_metadata

    i_train = self.df_metadata_train.index
    i_val = self.df_metadata_val.index
    i_test = self.df_metadata_test.index

    # set the train, val and test sets
    self.y_m_train = y_m[i_train]
    self.y_c_train = y_c[i_train]
    self.y_m_val = y_m[i_val]
    self.y_c_val = y_c[i_val]
    self.y_m_test = y_m[i_test]
    self.y_c_test = y_c[i_test]

    print('total samples  in training where y_m = 1: ', self.y_m_train.sum())
    print('total samples  in training where y_c = 1: ', self.y_c_train.sum())
    print('total samples  in training where y_m = 1 and y_c = 1: ', (self.y_m_train*self.y_c_train).sum())
    print('total samples  in training where y_m = 1 and y_c = 0: ', (self.y_m_train*(1-self.y_c_train)).sum())
    print('total samples  in training where y_m = 0 and y_c = 1: ', ((1-self.y_m_train)*self.y_c_train).sum())
    print('total samples  in training where y_m = 0 and y_c = 0: ', ((1-self.y_m_train)*(1-self.y_c_train)).sum())
    

  
  def create_sample(self, df_metadata, train_size, val_size, test_size, spurious_ratio, seed, p_m=False, set_X=False, combine_sets=True):
        """
        Create a (smaller) sample for the dataset, based on the train, val and test size
        """
        
        # select the training samples
        if combine_sets:

            # set X, y_m and y_c by concatenating the three sets
            if set_X:
                self.X = torch.cat([self.X_train, self.X_val, self.X_test])
            self.y_m = torch.cat([self.y_m_train, self.y_m_val, self.y_m_test])
            self.y_c = torch.cat([self.y_c_train, self.y_c_val, self.y_c_test])
            
            # make copies of X, y_m and y_c
            if set_X:
                X = self.X.clone()
            y_m = self.y_m.clone()
            y_c = self.y_c.clone()

            # set the sample split for train
            self.sample_train_ids = self.set_sample_split(y_m, y_c, train_size, spurious_ratio, p_m, seed)
            index = torch.ones(self.y_m.shape[0], dtype=bool)
            index[self.sample_train_ids] = False

            # remove the self.sample_train_ids indeces from X, y_m, y_c
            if set_X:
                X_after_train_selected = X[index,]
            y_m_after_train_selected = y_m[index]
            y_c_after_train_selected = y_c[index]
            
            

            # set the sample split for val
            self.sample_val_ids = self.set_sample_split(y_m_after_train_selected, y_c_after_train_selected, val_size, spurious_ratio, p_m, seed)
            index = torch.ones(y_m_after_train_selected.shape[0], dtype=bool)
            index[self.sample_val_ids] = False

            # remove the validation samples from the combined set
            if set_X:
                X_m_after_val_selected = X_after_train_selected[index,]
            y_m_after_val_selected = y_m_after_train_selected[index]
            y_c_after_val_selected = y_c_after_train_selected[index]

            # set the sample split for test
            self.sample_test_ids = self.set_sample_split(y_m_after_val_selected, y_c_after_val_selected, test_size, 0.5, p_m, seed)

            if set_X:
                self.X_train = self.X[self.sample_train_ids,]
                self.X_val = X_after_train_selected[self.sample_val_ids,]
                self.X_test = X_m_after_val_selected[self.sample_test_ids,]

            self.y_m_train = self.y_m[self.sample_train_ids]
            self.y_c_train = self.y_c[self.sample_train_ids]
            self.y_m_val = y_m_after_train_selected[self.sample_val_ids]
            self.y_c_val = y_c_after_train_selected[self.sample_val_ids]
            self.y_m_test = y_m_after_val_selected[self.sample_test_ids]
            self.y_c_test = y_c_after_val_selected[self.sample_test_ids]

        # if not combine sets, select from train/val/test separately
        else:
            self.sample_train_ids = self.set_sample_split(self.y_m_train, self.y_c_train, train_size, spurious_ratio, p_m, seed)
            self.sample_val_ids = self.set_sample_split(self.y_m_val, self.y_c_val, val_size, spurious_ratio, p_m, seed)
            self.sample_test_ids = self.set_sample_split(self.y_m_test, self.y_c_test, test_size, 0.5, p_m, seed)

            if set_X:
                self.X_train = self.X_train[self.sample_train_ids,]
                self.X_val = self.X_val[self.sample_val_ids,]
                self.X_test = self.X_test[self.sample_test_ids,]

            # set the attributes
            self.y_m_train = self.y_m_train[self.sample_train_ids]
            self.y_c_train = self.y_c_train[self.sample_train_ids]
            self.y_m_val = self.y_m_val[self.sample_val_ids]
            self.y_c_val = self.y_c_val[self.sample_val_ids]
            self.y_m_test = self.y_m_test[self.sample_test_ids]
            self.y_c_test = self.y_c_test[self.sample_test_ids]


        # change the self.df_metadata_train, self.df_metadata_val and self.df_metadata_test
        if df_metadata is not None:
            
            self.df_metadata_train = df_metadata.iloc[self.sample_train_ids.numpy()]

            index = torch.ones(self.y_m.shape[0], dtype=bool)
            index[self.sample_train_ids] = False

            df_metadata_after_train = df_metadata.iloc[index.numpy(),]
            self.df_metadata_val = df_metadata_after_train.iloc[self.sample_val_ids.numpy()]

            index = torch.ones(len(df_metadata_after_train.index), dtype=bool)
            index[self.sample_val_ids] = False

            df_metadata_after_val = df_metadata_after_train.iloc[index.numpy(),]
            self.df_metadata_test = df_metadata_after_val.iloc[self.sample_test_ids.numpy()]

            print('in df metadata train (main) ', self.df_metadata_train[self.main_task_name].head(5) )
            print('in df metadata val (main) ', self.df_metadata_val[self.main_task_name].head(5) )
            print('in y_m_train ', self.y_m_train[:5] )
            print('in y_m_val ', self.y_m_val[:5] )

            print('in df metadata train (concept) ', self.df_metadata_train[self.concept_name].head(5) )
            print('in df metadata val (concept) ', self.df_metadata_val[self.concept_name].head(5) )
            print('in y_c_train ', self.y_c_train[:5] )
            print('in y_c_val ', self.y_c_val[:5] )
          


        # sense check conditional probabilities
        print('In sample')
        print('Conditional probability of y_m = 1 given y_c = 1 in train: {}'.format(self.y_m_train[self.y_c_train==1].float().mean()))
        print('Conditional probability of y_m = 1 given y_c = 0 in train: {}'.format(self.y_m_train[self.y_c_train==0].float().mean()))
        print('Conditional probability of y_m = 0 given y_c = 1 in train: {}'.format((1-self.y_m_train)[self.y_c_train==1].float().mean()))
        print('Conditional probability of y_m = 0 given y_c = 0 in train: {}'.format((1-self.y_m_train)[self.y_c_train==0].float().mean()))


                
  def load_embeddings(self, folder_embeddings,  device, file_name= 'celebA_main_task_Blond_Hair_concept_Female_sample_n_images_all.pickle'):
        """
        Load the embeddings from a pickle file
        """


        with open(folder_embeddings +file_name, 'rb') as f:
                data_dict = pickle.load(f)

        self.X_train = data_dict['X_train'].to(device)
        self.X_val = data_dict['X_val'].to(device)
        self.X_test = data_dict['X_test'].to(device)
        self.y_m_train = data_dict['y_m_train'].to(device)
        self.y_m_val = data_dict['y_m_val'].to(device)
        self.y_m_test = data_dict['y_m_test'].to(device)
        self.y_c_train = data_dict['y_c_train'].to(device)
        self.y_c_val = data_dict['y_c_val'].to(device)
        self.y_c_test = data_dict['y_c_test'].to(device)

        # print the shapes of all the tensors
        print('X_train shape: {}'.format(self.X_train.shape))
        print('X_val shape: {}'.format(self.X_val.shape))
        print('X_test shape: {}'.format(self.X_test.shape))
        print('y_m_train shape: {}'.format(self.y_m_train.shape))
        print('y_m_val shape: {}'.format(self.y_m_val.shape))
        print('y_m_test shape: {}'.format(self.y_m_test.shape))
        print('y_c_train shape: {}'.format(self.y_c_train.shape))
        print('y_c_val shape: {}'.format(self.y_c_val.shape))
        print('y_c_test shape: {}'.format(self.y_c_test.shape))
        


  
  def set_dataset(self, df_meta_data, embeddings, main_task_name='Blond_hair', concept_name='Male', set_X= False):
    """
    Function to set the dataset, based on the df_meta_data and embeddings
    """
      
    # get the main-task and concept labels
    y_m = torch.tensor(df_meta_data[main_task_name].replace({-1:0}).values)
    y_c = torch.tensor(df_meta_data[concept_name].replace({-1:0}).values)

    # get indices for train, val and test from df_metadata
    self.df_metadata_train = df_meta_data[df_meta_data['split'] == 0]
    self.df_metadata_val = df_meta_data[df_meta_data['split'] == 1]
    self.df_metadata_test = df_meta_data[df_meta_data['split'] == 2]

    # set the attributes
    i_train = self.df_metadata_train.index
    i_val = self.df_metadata_val.index
    i_test = self.df_metadata_test.index

    # set the train, val and test sets
    self.y_m_train = y_m[i_train]
    self.y_c_train = y_c[i_train]
    self.y_m_val = y_m[i_val]
    self.y_c_val = y_c[i_val]
    self.y_m_test = y_m[i_test]
    self.y_c_test = y_c[i_test]

    if set_X:
        self.X_train = embeddings[i_train, ]
        self.X_val = embeddings[i_val, ]
        self.X_test = embeddings[i_test, ]


    print('total samples  in training where y_m = 1: ', self.y_m_train.sum())
    print('total samples  in training where y_c = 1: ', self.y_c_train.sum())
    print('total samples  in training where y_m = 1 and y_c = 1: ', (self.y_m_train*self.y_c_train).sum())
    print('total samples  in training where y_m = 1 and y_c = 0: ', (self.y_m_train*(1-self.y_c_train)).sum())
    print('total samples  in training where y_m = 0 and y_c = 1: ', ((1-self.y_m_train)*self.y_c_train).sum())
    print('total samples  in training where y_m = 0 and y_c = 0: ', ((1-self.y_m_train)*(1-self.y_c_train)).sum())
    
      
    
    

  def load_dataset(self, transformation_func, main_task_name='Eyeglasses', concept_name='Smiling', sample_n_images=1000, spurious_ratio = 0.5, train_size=800, val_size=100, test_size=100, seed=12345, load_full_images=False, create_embeddings=True, folder='raw_input/', device_type='mps', p_m=0.5, finetuned=False, settings=None, adversarial=True, early_stopping=True, balance_main=True, device=None):
    """
    Function to load the dataset (either original images, embeddings)
    """

    set_seed(seed)

    # set the main task name and concept name
    self.set_dependent_var( main_task_name, concept_name, folder)
      
    
    # create sample
    if sample_n_images is not None:

        # create sample
        self.create_sample(df_metadata=self.df_metadata, train_size=train_size, val_size=val_size, test_size=test_size, spurious_ratio=spurious_ratio, seed=seed, p_m=p_m)

    # load the original full images
    if load_full_images:

            celebA_images_chunk_train, celebA_labels_chunk_train = load_images(self.df_metadata_train, transformation_func, folder+'images', '')
            celebA_images_chunk_val, celebA_labels_chunk_val = load_images(self.df_metadata_val, transformation_func, folder+'images', '')
            celebA_images_chunk_test, celebA_labels_chunk_test = load_images(self.df_metadata_test, transformation_func, folder+'images', '')

            celebA_images_chunk_train = celebA_images_chunk_train.squeeze(-1)
            celebA_images_chunk_val = celebA_images_chunk_val.squeeze(-1)
            celebA_images_chunk_test = celebA_images_chunk_test.squeeze(-1)

            # set the train, val and test sets
            self.X_train = celebA_images_chunk_train
            self.X_val = celebA_images_chunk_val
            self.X_test =celebA_images_chunk_test

    # create embeddings
    if create_embeddings:
            device = torch.device(device_type)

            model = torchvision.models.resnet50(pretrained = True)
            embedding_model = embedding_creator(model, prevent_finetuning=True)
            embedding_model = embedding_model.to(device)
            # set the model to evaluation mode
            embedding_model = embedding_model.eval()

            if finetuned:

                    
                state_dict = torch.load('finetuned_models/resnet50_model_finetuned_celebA_settings_{}_spurious_ratio_{}_balance_main_{}_early_stopping_{}_adversarial_{}_seed_{}.pt'.format(settings, int(100*spurious_ratio), int(balance_main), int(early_stopping), int(adversarial), seed))
                embedding_model.load_state_dict(state_dict)
                print('Loaded finetuned model')

            celebA_embedding_chunk_train, celebA_labels_chunk_train = self.get_embeddings_in_chunks(10000, self.df_metadata_train, transformation_func, folder+'images', device, embedding_model)
            celebA_embedding_chunk_val,  celebA_labels_chunk_val = self.get_embeddings_in_chunks(10000, self.df_metadata_val,  transformation_func, folder+'images', device, embedding_model)
            celebA_embedding_chunk_test,  celebA_labels_chunk_test = self.get_embeddings_in_chunks(10000, self.df_metadata_test, transformation_func, folder+'images', device, embedding_model)

            # set the train, val and test sets
            self.X_train = celebA_embedding_chunk_train
            self.X_val = celebA_embedding_chunk_val
            self.X_test =celebA_embedding_chunk_test

           
            


def get_dataset_obj(dataset, dataset_settings, spurious_ratio, data_info, seed, device, adversarial=False, use_punctuation_MNLI=False):
        """
        prepare the dataset object for the given dataset
        """

        # if Waterbirds dataset
        if dataset == 'Waterbirds':

            # create object
            data_obj = Waterbird_Dataset()

            # load dataset for training and validation (spurious ratio) and the test set
            spurious_ratio_int = int(100*spurious_ratio)
            dataset_folder = 'datasets'
            directory_labels = dataset_folder + '/' +'Waterbirds/images/waterbirds_{}/tensors'.format(spurious_ratio_int)
            directory_embeddings = dataset_folder + '/' +'Waterbirds/embeddings/waterbirds_{}_{}'.format(spurious_ratio_int, dataset_settings['model_name'])
            directory_labels_test = dataset_folder + '/' +'Waterbirds/images/waterbirds_50/tensors'
            directory_embeddings_test =dataset_folder + '/' + 'Waterbirds/embeddings/waterbirds_50_{}'.format(dataset_settings['model_name'])
            
            # load metadata
            directory_df_meta = dataset_folder + '/' + 'Waterbirds/images/waterbirds_{}/metadata_waterbird_{}_{}_50.csv'.format(spurious_ratio_int, spurious_ratio_int, spurious_ratio_int)
            directory_metadata_test = dataset_folder + '/' +'Waterbirds/images/waterbirds_50/metadata_waterbird_50_50_50.csv'
            df_meta = pd.read_csv(directory_df_meta)
            df_meta_test = pd.read_csv(directory_metadata_test)


            # load dataset
            data_obj.load_dataset(spurious_ratio_int, directory_labels, directory_embeddings, directory_labels_test, directory_embeddings_test, df_meta, df_meta_test , device=device, model_name=dataset_settings['model_name'], image_type_train_val=dataset_settings['image_type_train_val'], image_type_test=dataset_settings['image_type_test'], corr_test=dataset_settings['corr_test'], adversarial=adversarial, seed=seed, balance_main=dataset_settings['balance_main'])


        # if multiNLI dataset
        elif dataset == 'multiNLI':
            data_obj = multiNLI_Dataset()

            folder_data = 'datasets/multiNLI/embeddings'

            if dataset_settings['spurious_ratio_train'] is not None:
                spurious_ratio_sample = spurious_ratio
                spurious_ratio_train = dataset_settings['spurious_ratio_train']
                filename_data = folder_data + '/'+  'MNLI_data_settings_{}_embedding_type_{}_finetuned_{}_train_size_{}_val_size_{}_spurious_ratio_train_{}_spurious_ratio_sample_{}_binary_{}_dropout_{}_early_stopping_{}_finetune_mode_{}_seed_{}'.format( dataset_settings['finetune_param_type'], dataset_settings['embedding_type'], int(dataset_settings['finetuned_BERT']), dataset_settings['train_size'], dataset_settings['val_size'], int(100*spurious_ratio_train), int(100*spurious_ratio_sample), int(dataset_settings['binary_task']), int(100*dataset_settings['dropout']), int(dataset_settings['early_stopping']),dataset_settings['finetune_mode'], seed)
                
            else:
                filename_data = folder_data + '/'+  'MNLI_data_settings_{}_embedding_type_{}_finetuned_{}_train_size_{}_val_size_{}_spurious_ratio_train_{}_spurious_ratio_sample_{}_binary_{}_dropout_{}_early_stopping_{}_finetune_mode_{}_seed_{}'.format( dataset_settings['finetune_param_type'], dataset_settings['embedding_type'], int(dataset_settings['finetuned_BERT']), dataset_settings['train_size'], dataset_settings['val_size'], int(100*spurious_ratio), int(100*spurious_ratio), int(dataset_settings['binary_task']), int(100*dataset_settings['dropout']), int(dataset_settings['early_stopping']),dataset_settings['finetune_mode'], seed)

            print('File loaded: ', filename_data)
                

            if use_punctuation_MNLI:
                filename_data = filename_data + '_punctuation'
            filename_data +='.pkl'

            # load the embeddings 
            data_obj.load_processed_dataset(filename_data, device=device)

        # if celebA dataset
        elif dataset =='celebA':
            data_obj = celebA_Dataset()


            if dataset_settings['adversarial']:
                filename_data  = 'celebA_main_task_' + dataset_settings['main_task_name'] + '_concept_' + dataset_settings['concept_name'] + '_sample_n_images_' +str(dataset_settings['train_size']) + '_finetuned_settings_' + dataset_settings['settings'] + '_spurious_ratio_' + str(int(spurious_ratio*100))+ '_early_stopping_' + str(int(dataset_settings['early_stopping']))+  '_adversarial_'  + str(int(dataset_settings['adversarial']))

            else:
                filename_data = 'celebA_main_task_Blond_Hair_concept_Female_sample_n_images_all.pickle'
                data_obj.set_dependent_var(dataset_settings['main_task_name'], dataset_settings['concept_name'],  folder='datasets/celebA/raw_input/' )

            # load the embeddings
            data_obj.load_embeddings( 'datasets/celebA/embeddings/',  device, file_name= filename_data)

            if dataset_settings['set_sample']:

                data_obj.create_sample(df_metadata=None, 
                                       train_size=dataset_settings['train_size'], 
                                       val_size=dataset_settings['val_size'], 
                                       test_size=dataset_settings['test_size'],
                                       spurious_ratio=spurious_ratio,
                                       seed=seed,
                                       p_m=dataset_settings['p_m'],
                                       set_X=True,
                                       combine_sets=dataset_settings['combine_sets'],
                                       )
                print('created sample')

        # if Toy dataset
        elif dataset == 'Toy':
            data_obj = Toy_Dataset()

            # load dataset for training and validation
            set_seed(seed)
            X, y_c, y_m, sigma, = data_obj.sample_data(n=dataset_settings['n'], 
                                    d=dataset_settings['d'],
                                    rho_c = dataset_settings['rho_c'],
                                    rho_m = dataset_settings['rho_m'],
                                    gamma_c=dataset_settings['gamma_c'],
                                    gamma_m=dataset_settings['gamma_m'],
                                    intercept_concept=dataset_settings['intercept_concept'],
                                    intercept_main=dataset_settings['intercept_main'],
                                    X_variance=dataset_settings['X_variance'],
                                    angle=dataset_settings['angle'],
                                    rho_c_m=spurious_ratio)
            # set the training/validation split
            data_obj.set_train_val_split(seed=seed, train_split=data_info['All']['train_split'])
            

            print('train size: ', data_obj.X_train.shape[0])
            print('val size: ', data_obj.X_val.shape[0])
            print('Sigma: ', sigma)
            

            set_seed(seed)
        
            # set the test set - no correlation between the features
            X_test, y_c_test, y_m_test, sigma_test=  data_obj.sample_data(n=dataset_settings['n'], 
                                    d=dataset_settings['d'],
                                    rho_c = dataset_settings['rho_c'],
                                    rho_m = dataset_settings['rho_m'],
                                    gamma_c=dataset_settings['gamma_c'],
                                    gamma_m=dataset_settings['gamma_m'],
                                    intercept_concept=dataset_settings['intercept_concept'],
                                    intercept_main=dataset_settings['intercept_main'],
                                    X_variance=dataset_settings['X_variance'],
                                    angle=dataset_settings['angle'],
                                    rho_c_m=0) # no correlation between the features
            
            # set the test set as part of the dataset object
            data_obj.X_test = X_test
            data_obj.y_c_test = y_c_test
            data_obj.y_m_test = y_m_test
        
        return data_obj




def get_group_weights(y_c, y_m):
    """
    Get the weights for each group
    """
    groups = Dataset_obj.get_groups(y_c=y_c, y_m=y_m)
    n_per_group = Dataset_obj.get_n_per_class(torch.Tensor(groups).int(), classes= [0, 1,2,3])
    weights = Dataset_obj.give_balanced_weights(groups, len(groups), n_per_group, normalized=False).unsqueeze(-1)

    return weights



def torch_calc_BCE_weighted(y_m, y_c, y_pred, device, reduce='none', type_dependent='main'):
     
     """
     go over each group, calculate BCE, then give average
     """

     groups = Dataset_obj.get_groups(y_c=y_c, y_m=y_m)
     n_per_group = Dataset_obj.get_n_per_class(torch.Tensor(groups).int(), classes= [0, 1,2,3])
     weights = Dataset_obj.give_balanced_weights(groups, len(groups), n_per_group, normalized=False).unsqueeze(-1)


     # calculate the BCE for each group
     if type_dependent == 'main':
         BCE_weighted = torch_calc_BCE(y_m, y_pred, device, reduce=reduce, weighted = True, weights=weights)
     elif type_dependent == 'concept':
         BCE_weighted = torch_calc_BCE(y_c, y_pred, device, reduce=reduce, weighted = True, weights=weights)
     

   
     return BCE_weighted
     
          
     
    

def torch_calc_BCE(y, y_pred, device, reduce='mean', weighted = False, weights=None):
    """
    Calculates the binary cross entropy loss for a given y and y_pred
    """

    if weighted and weights is None:
        weights = get_class_weights(y)
    
    if not weighted:
        weights = None

    return nn.BCEWithLogitsLoss(reduction=reduce, weight=weights)(y_pred, y.unsqueeze(1).float().to(device))

