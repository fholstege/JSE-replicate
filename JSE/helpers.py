
import torch.nn as nn
import torch
import numpy as np
import random
import pandas as pd
from PIL import Image
import numpy as np
import numpy as np
from torchvision import transforms
import torch 
from torch.utils.data import TensorDataset
import os
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import sys
import matplotlib.pyplot as plt


def return_group(y_c, y_m):
    """
    Return the group of a sample based on y_c and y_m 
    """
    if y_c == 0 and y_m == 0:
        group = 0
    elif y_c == 0 and y_m == 1:
        group = 1
    elif y_c == 1 and y_m == 0:
        group = 2
    elif y_c ==1 and y_m == 1:
        group = 3

    return group

def str_to_bool(s):
    """
    Converts a string to a boolean
    """
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was


def get_class_weights(y):
    """
    Get the class weights for a given y
    """
    n_class_0 = (y == 0).sum()
    n_class_1 = (y == 1).sum()
    n = len(y)
    weight_class_1 = 1/(n_class_1/n)
    weight_class_0 = 1/(n_class_0/n)
    weight_class_1_norm = weight_class_1/(weight_class_1 + weight_class_0)
    weight_class_0_norm = weight_class_0/(weight_class_1 + weight_class_0)
    weights = torch.where(y == 1, weight_class_1_norm, weight_class_0_norm).unsqueeze(-1)

    return weights


def turn_output_class(y_pred):
    return torch.round(torch.sigmoid(y_pred))


def get_weighted_acc_pytorch_model(y_m, y_pred, y_c, turn_classes = True):
     """
     Calculates the accuracy for a given y_m and y_pred (weighted by the concept)

     """
    
     # get the accuracy per group 
     acc_per_group, _ = get_acc_per_group(y_pred, y_m, y_c, y_z=None, turn_classes=turn_classes)
     print('acc per group', acc_per_group)

     # take the average over each of the accuracies
     weighted_acc = acc_per_group.mean().values[1]

     return weighted_acc

def get_acc_pytorch_model(y, y_pred,  turn_classes = True, weighted_accuracy = False, weights=None):
    """
    Calculates the accuracy for a given y and y_pred
    """
    
    if turn_classes:
        classes = turn_output_class(y_pred)
        
    else:
        classes = y_pred
    classes = classes.to(device=y.device)        

    acc =  (y.unsqueeze(-1) == classes).sum()/len(y)

    return acc

def get_acc_per_group(y_pred, y_m, y_c, y_z=None, turn_classes=True):
    """
    calculates accuracy per group
    """

    if turn_classes:
         classes = turn_output_class(y_pred)
    else:
        classes = y_pred

    correct = (y_m.unsqueeze(-1) == classes).int()
    
    # if also third concept is present
    if y_z is not None:
        results = torch.hstack((y_m.detach().unsqueeze(-1), classes.detach(), y_c.detach().unsqueeze(-1), y_z.detach().unsqueeze(-1), correct.detach().float()))
    else:
        results = torch.hstack((y_m.detach().unsqueeze(-1), classes.detach(), y_c.detach().unsqueeze(-1), correct.detach().float()))

    df_results = pd.DataFrame(results.cpu().numpy())

    if y_z is not None:
        df_results.columns = ['main', 'pred', 'concept', 'alt_concept', 'correct']
    else:
        df_results.columns = ['main', 'pred', 'concept', 'correct']
    
    if y_z is not None:
        df_result_per_group = df_results.groupby(['main', 'concept', 'alt_concept'])['correct'].agg(['count','mean'])
    else:
        df_result_per_group = df_results.groupby(['main', 'concept'])['correct'].agg(['count','mean'])

    avg_correct = correct.float().mean()
    
    return df_result_per_group, avg_correct

def set_seed(seed):
    """
    Sets the seed for the random number generators
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_random_predictions(p, n):
    """
    Returns a tensor of random predictions with probability p
    """

    p_pred = torch.Tensor([p]*n)

    return p_pred


def get_acc_random(y, p, n, weighted=False, y_c=None):
    """
    Returns the accuracy of random predictions with probability p
    """
    random_pred = get_random_predictions(p, n).unsqueeze(1).float()
    classes_pred = torch.round(random_pred)

    # calculate accuracy to print for indication
    if weighted:
        acc_random = get_weighted_acc_pytorch_model(y, classes_pred, y_c, turn_classes = False)
    else:
        acc_random = get_acc_pytorch_model(y, classes_pred, turn_classes = False)

    return acc_random



def Convert(string, type_=float):
        
    """
    Turn string to list of certain objects
    """
    li = list(string.split("-"))

    li_float = [type_(x) for x in li]

    return li_float

def add_info_df(df, sim_i, corr_concept_main):
    """
    Add information to the dataframe
    """

    df.reset_index(inplace=True)
    df['sim'] = sim_i
    df['corr_concept_main'] = corr_concept_main


def make_accuracy_main_plot(df_result, sims, title='Our method', models = ['joint', 'rafvogel', 'ERM'], labels = ['joint', 'rafvogel', 'ERM'], linestyle='-.', colors = ['blue', 'red', 'green'], ylabel='Accuracy on main task'):
    """
    Make the accuracy plot for the main task
    """
            
    for model in models:

        df_result_method = df_result[df_result['model'] == model]
        color = colors[models.index(model)]
        label = labels[models.index(model)]

        plt.errorbar(df_result_method['corr_concept_main'], 
                            df_result_method['mean'], 
                            yerr = df_result_method['std']/np.sqrt(sims),
                            label = label,
                            linestyle = linestyle,
                            color =color)
                

        plt.xlabel('Correlation between main and concept')
        plt.ylabel(ylabel)


def get_linear_decision_boundary(v,  steps=100, b=0, xmin=-4, xmax=4):
    """
    Get the linear decision boundary
    """

    x1_span = np.linspace(xmin, xmax, steps)
    
    a = -v[0] / v[1]
    
    decision_boundary = a * x1_span - (b) / v[1]
    
    return decision_boundary, x1_span

def get_linear_combination(v, xmin, xmax, steps=100, b=0, X_index=0):
     """
     Get a linear combination of the weights
     """
     
     x_span = np.linspace(xmin, xmax, steps)

     a = v[1] / v[0]

     linear_combination = a * x_span - (b) / v[0]

     return linear_combination, x_span


def get_worst_group_acc(groups, y_m, y_m_pred):
    """
    get the worst group accuracy
    """

    acc_group_list = []
    for group in [1, 2, 3, 4]:
        group_sample = groups == group

        y_m_group = y_m[group_sample]
        y_m_pred_group = y_m_pred[group_sample]

        acc_group = get_acc_pytorch_model(y_m_group, y_m_pred_group, turn_classes=True)
        acc_group_list.append(acc_group)

        print('group: {}, acc: {}'.format(group, acc_group))

    return min(acc_group_list)


def load_images(df_metadata, transform_func, base_dir, type_img):
    
     """
        Loads images from a dataframe with metadata
     """
     
     # get the full directory where the images are stored
     full_dir = base_dir + type_img + '/'
    
     # gather image names
     image_paths = df_metadata['img_filename']
    
     # create list to be filled with image tensors
     list_image_tensors = [None]*len(image_paths)
    
     # the dependent variable 
     y_arr = df_metadata['y'].values
     
     # go over each image and load
     i = 0
     for image_path in image_paths:
         
       # list image file path
       full_image_path = full_dir + image_path
    
       if os.path.isfile(full_image_path):
               
            # load and convert to RGB
            img = Image.open(full_image_path).convert('RGB')
        
            # transform image according to specified function
            img_transformed = transform_func(img)
        
            # add said tensor to list
            list_image_tensors[i] = img_transformed
    
       else:
            # print if not available to notify user  
            list_image_tensors = list_image_tensors[-i]
            print('Not available: {}'.format(full_dir + image_path))
           
       # print progress
       i += 1
       if i% 1000 == 0:
            print('Loading image at: {}/{}'.format(i, len(image_paths)))
     
     # combine list of images and array of labels to tensor dataset
     tensor_images, tensor_labels = turn_list_images_labels_to_tensor(list_image_tensors, y_arr)
     
     return tensor_images, tensor_labels


def turn_list_images_labels_to_tensor(list_images, labels, images_array = False, to_tensor_dataset = False):
  """
  Turns a list of images and labels to a tensor dataset
  """  

  if images_array:
      
      # turn array to tensors via this function
      to_tensor_func = transforms.Compose([
            transforms.ToTensor()
        ])
      
      # turn images to tensor
      tensor_images = torch.stack([to_tensor_func(img) for img in list_images])
  else:
      tensor_images = torch.stack(list_images)
      
  
      
  tensor_labels =  torch.from_numpy(labels).int()
  
  if to_tensor_dataset:
      # combine in tensor dataset
      tensor_dataset = TensorDataset(tensor_images, tensor_labels)
      
      return tensor_dataset
  else:
      return tensor_images, tensor_labels