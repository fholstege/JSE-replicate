#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:56:42 2023

"""
import torch.nn as nn
import mctorch.nn as mnn
import torch
from torch.autograd import Variable
import sys

def create_P(v, use_inverse = True):
    """
    Creates projection matrix P; P = v v^T
    """
    
    # # v transpose v, and its inverse
    if use_inverse:
      v_t_v = torch.matmul(v.T, v)
  

      #v_t_v_inv = torch.linalg.pinv(v_t_v)
      v_t_v_inv = torch.linalg.inv(v_t_v)

      P = torch.matmul(v, torch.matmul(v_t_v_inv, v.T)).to(v.device)

    else:
      
      # create projection matrix
      v_unit = v/torch.norm(v, p=2)
      P = torch.matmul(v_unit, v_unit.T).to(v.device)

    
    return P

def init_zero_weights(module, bias=True, eps=0.0000001):
    """
    Initialize weights of a linear layer to zero
    """
    
    if isinstance(module, nn.Linear):
          
          module.weight.data.fill_(0 + eps)
          
          if bias:
              module.bias.data.fill_(0+eps) 


class linear_model(nn.Module):
    """
    Linear model that takes the output of the feature extractor as input
    """
    
    
    def __init__(self, feature_size, output_size, bias=True):
        super(linear_model, self).__init__()
    
    
        # set the last layer to be a linear layer that takes the output of feature extractor
        self.linear_layer = nn.Linear(feature_size, output_size, bias) 
            
        
     
    def forward(self, x):
    
        output = self.linear_layer(x)
    
        return output
              

class P_orth(nn.Module):
    """
    nn.Module that projects the input x onto the orthogonal complement of v, where v is a unit vector
    """
    
    def __init__(self, v, device):
      super(P_orth, self).__init__()
      
      self.device = device
      self.v = v.to(self.device)
      self.I =  torch.eye(self.v.shape[0]).to(self.device)

      
     
    
    def forward(self,x ):
        
        # create projection matrix
        self.P = create_P(self.v, use_inverse=True).to(self.device)
        
        
        # create orthogonal projection matrix
        self.P_orth =  (self.I - self.P).to(self.device)

        # project x onto orthogonal complement of v
        x_P_orth = torch.matmul(x, self.P_orth).to(self.v.device)
        
       
        return x_P_orth
    
class joint_model(nn.Module):


    def __init__(self, d, device, bias=False):
      """
      Estimates two logistic regressions, of which the coefficients are orthogonal - using the stiefel manifold
      """
      super(joint_model, self).__init__()
      self.device = device

      # defines the coefficients for the concept (first column) and main-task (second column)
      self.W = nn.Linear(in_features=d, out_features=2, bias = True) #mnn.rLinear(in_features=d, out_features=2, weight_manifold=mnn.Stiefel, bias = bias)
      self.w_c = nn.Linear(in_features = 1, out_features = 1, bias = False)
      self.w_m = nn.Linear(in_features = 1, out_features = 1, bias = False)

    
    def return_coef(self, type_vec = 'main'):
       
      W_matrix = self.W.weight.data.T

      # get the concept and main-task coefficients

      if type_vec == 'main':
        w_m = W_matrix[:,1]
        #v_m = w_m/torch.norm(w_m, p=2)
        return w_m
      
      elif type_vec == 'concept':
        w_c = W_matrix[:,0].T
        #v_c = w_c/torch.norm(w_c, p=2)
        return w_c
    

    def forward(self, x):
       
      # get output for the concept and main-task
      output_concept = torch.matmul(x, (self.W.weight.T[:,0]* self.w_c.weight).T) + self.W.bias[0]
      output_main =torch.matmul(x, (self.W.weight.T[:,1]* self.w_m.weight).T) + self.W.bias[1]

      # put both in dict
      output_dict = {'y_c_1': output_concept.squeeze(-1), 'y_m_1':output_main.squeeze(-1)}
     

      return output_dict
    


  


    
              
class joint_main_concept_model(nn.Module):
      """
        Estimates a concept and main-task vector that are orthogonal to each other
      """
    
    
      def __init__(self, d, device,  bias=False):
        super(joint_main_concept_model, self).__init__()

        self.device = device
    
        # defines the coefficients for the concept
        self.w_c= nn.Linear(d, 1, bias, device=self.device) 
        
        # extract the coefficients
        self.w_c_coef =self.w_c.weight.T.to(self.device)
        
        # create the projection matrix based on coefficients for concept
        self.P_w_c = P_orth(self.w_c_coef, device = self.device)
        
        
        # defines the main-task coefficients
        self.w_m = nn.Sequential(
                self.P_w_c, # before data is passed into it, apply projection matrix
                nn.Linear(d, 1, bias, device=self.device) 
                )
        
        # extract the coefficients for the main-task
        self.w_m_coef = torch.zeros((d, 1)).to(self.device) # filled with zeros, need to apply projection matrix before extracting
        

      def forward(self, x):
        
        # set all the vectors
        self.w_c_coef =self.w_c.weight.T
        self.P_w_c = P_orth(self.w_c_coef, self.device)
        self.w_m_coef = self.P_w_c(self.w_m[1].weight).T
        
        # get output for the concept
        y_c = self.w_c(x)
        
        # get the output for the main-task
        y_m = self.w_m(x)
        
        # put both in dict
        output_dict = {'y_c_1': y_c, 'y_m_1':y_m }
       
    
        return output_dict
    
      # function to return the concept/main-task vectors
      def return_coef(self, type_vec = 'main'):
          
          if type_vec =='main':
              w_m = self.P_w_c(self.w_m[1].weight).T
              
              return w_m
          
          elif type_vec=='concept':
              w_c =self.w_c.weight.T
              
              return w_c



####### embeddings
class embedding_creator(nn.Module):
  """
  Create version of the model, specifically for creating embeddings
  """

  def __init__(self, original_resnet, prevent_finetuning = True):
    super(embedding_creator, self).__init__()

    # set require_grad = False to all the original resnet layers
    if prevent_finetuning:
      for param in original_resnet.parameters(): # this ensures these layers are not trained subsequently
        param.requires_grad = False

    # select everything but the last layer
    self.list_feature_modules = list(original_resnet.children())[:-1]

    # define the feature extractor 
    self.feature_extractor =  nn.Sequential(
                    # stop at conv4
                    *self.list_feature_modules
                )
    
    self.flatten = nn.Flatten(start_dim =1, end_dim= -1)
    
  def forward(self, x):

    # get embedding
    embedding = self.feature_extractor(x)
    embedding_flat = self.flatten(embedding)

    return embedding_flat

def get_embedding_in_batches( model, loader):

  # total steps to take
  total_step_val = len(loader)

  model = model.eval()   # Set model to evaluate mode
  list_embedding_batches = [None]*total_step_val

  # go over each batch
  for i, (images) in enumerate(loader):
     
    # input and output batch
    b_x =Variable(images[0])

    # get output and calc acc
    embedding = model(b_x)
    
    list_embedding_batches[i] = embedding
    
    print('Step {}/{}'.format(i+1, total_step_val))
    
  print('Done gathering')
  if len(list_embedding_batches) == 1:
    all_embeddings = list_embedding_batches[0]
    print('at this one, shape is', all_embeddings.shape)

  else:
    embeddings = torch.stack(list_embedding_batches[:-1], dim=0).flatten(start_dim=0, end_dim=1)
  
    last_embedding = list_embedding_batches[-1]
    all_embeddings = torch.cat((embeddings, last_embedding), dim = 0)
    print('usually, shape is', all_embeddings.shape)
    

  return all_embeddings
  

