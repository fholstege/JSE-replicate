#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:59:03 2023

"""

import math
import datetime
from transformers import get_scheduler

import mctorch.optim as moptim

import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from JSE.helpers import *
from JSE.models import *
from JSE.data import *

import scipy.stats as stats
import numpy as np

import time
import sys 


def group_weighted_d_mean_estimator(g, d):
    """
    Get the group weighted mean estimator for d
    """

    # check if two dimensional tensor - if yes, turn one dimensional
    if len(d.shape) == 2:
        d = d.squeeze()
    

    # group the observations in d according to the group they belong to
    d_1 = d[g==1]
    d_2 = d[g==2]
    d_3 = d[g==3]
    d_4 = d[g==4]


    # calculate the mean of each group
    mean_d_1 = d_1.mean()
    mean_d_2 = d_2.mean()
    mean_d_3 = d_3.mean()
    mean_d_4 = d_4.mean()



    # calculate the weighted mean
    mean_est = (1/4)*(mean_d_1 + mean_d_2 +mean_d_3 + mean_d_4)
   

    return mean_est


def get_different_vars(g, d):
    """       
    Get the variance of d for each group
    """
      
    # group the observations in d according to the group they belong to
    d_1 = d[g==1]
    d_2 = d[g==2]
    d_3 = d[g==3]
    d_4 = d[g==4]

    # calculate the variance of each group
    var_d_1 = d_1.var(correction=1)
    var_d_2 = d_2.var(correction=1)
    var_d_3 = d_3.var(correction=1)
    var_d_4 = d_4.var(correction=1)

    return var_d_1, var_d_2, var_d_3, var_d_4



def calc_variance_estimator(n, sigma_1, sigma_2, sigma_3, sigma_4, n_1, n_2, n_3, n_4):
     """
     Combine the variances of the different groups to get the variance estimator
     """
     
     constant = (1/16)

     var_est = constant * ((sigma_1/n_1+  sigma_2/n_2 + sigma_3/n_3 +  sigma_4/n_4))

     return var_est


def calc_t_stat(mean_est, var_est, delta):
    """
    Calculate the t-statistic
    """
    std_est = np.sqrt(var_est)
         
    t_stat = (mean_est - delta)/std_est
    
    return t_stat

def get_weighted_BCE_diff_t_stat(g, bce_diff, delta, n):
      """
      Get the t-statistic for the difference in BCE loss between the concept and main task
      """
   
      group_weighted_mean_est = group_weighted_d_mean_estimator(g, bce_diff)
      var_1_est, var_2_est, var_3_est, var_4_est = get_different_vars(g, bce_diff)

      # get size of each group via g
      n_1 = g[g==1].shape[0]
      n_2 = g[g==2].shape[0]
      n_3 = g[g==3].shape[0]
      n_4 = g[g==4].shape[0]

      variance_estimator = calc_variance_estimator(n, var_1_est, var_2_est, var_3_est, var_4_est, n_1, n_2, n_3, n_4)
      t_stat = calc_t_stat(group_weighted_mean_est, variance_estimator, delta)

      return t_stat


def get_weighted_BCE_diff_t_stat_compare(y_m, y_c, y_pred_m, y_pred_c,g, delta, n, device):
    """
    Get the weighted t-statistic for the difference in BCE loss between the concept and main task
    """

    BCE_y_m = torch_calc_BCE(y_m, y_pred_m, device, reduce='none', weighted=False) 
    BCE_y_c = torch_calc_BCE(y_c, y_pred_c, device, reduce='none', weighted=False) 
    BCE_diff = (BCE_y_c - BCE_y_m)
    t_stat = get_weighted_BCE_diff_t_stat(g, BCE_diff, delta, n)

    return t_stat

def get_BCE_diff_random_classifier(y, y_pred, p):
    """
    Get the difference in BCE loss between the predictions and random predictions
    """
    n = len(y)

    random_pred = get_random_predictions(p, n).unsqueeze(1).float().to(device=y.device)
    p_y_pred = torch.sigmoid(y_pred).to(device=y.device)
    
    # calculate the binary cross entropy loss
    BCE_func =  nn.BCELoss(reduction='none', weight=None)
    BCE_random = BCE_func(random_pred, y.unsqueeze(-1).float()) 
    BCE_y_pred = BCE_func(p_y_pred, y.unsqueeze(-1).float())

    print('BCE random: {}'.format(BCE_random.mean()))
    print('BCE trained: {}'.format(BCE_y_pred.mean()))
    
    # calculate the difference in binary cross entropy loss and its mean + std
    BCE_diff =   (BCE_y_pred - BCE_random) 

    return BCE_diff


def get_weighted_BCE_diff_t_stat_random(y, y_pred, p, g, delta, n):
    """
    Get the weighted t-statistic for the difference in BCE loss between the predictions and random predictions
    """

    BCE_diff = get_BCE_diff_random_classifier(y, y_pred, p)

    t_stat = get_weighted_BCE_diff_t_stat(g, BCE_diff, delta, n)

    return t_stat
        

def calc_BCE_diff_random(y_pred, y, p, adjust = 1, weighted=False, weight=None):
    """
    Calculates the difference in binary cross entropy loss between the predictions and random predictions
    """

    BCE_diff = get_BCE_diff_random_classifier(y, y_pred, p)
    mean_BCE_diff = BCE_diff.mean()
    std_BCE_diff = BCE_diff.std() 

    
    return mean_BCE_diff, std_BCE_diff



def calc_BCE_difference_stats(y_pred_c,y_pred_m, y_c, y_m, device):
    """
    Calculates the difference in binary cross entropy loss between the predictions for the concept and main task
    Then calculates the mean and standard deviation of the difference
    """

    BCE_y_m = torch_calc_BCE(y_m, y_pred_m, device, reduce='none', weighted=False) 
    BCE_y_c = torch_calc_BCE(y_c, y_pred_c, device, reduce='none', weighted=False) 
        

    BCE_diff =  (BCE_y_c - BCE_y_m)
    print('BCE diff (concept/main): {}'.format(BCE_diff.mean()))

    mean_BCE_diff = BCE_diff.mean()

        
    std_BCE_diff = BCE_diff.std() 

    return mean_BCE_diff, std_BCE_diff
    


def get_baseline_diff(data_obj, loaders, device,  solver, lr,  per_step, tol, early_stopping, patience, epochs, bias=False, model_name='baseline', weight_decay=0.0, eval_balanced=False, return_model_results=False, use_training_data_for_testing=False):
    
    """
    Get the baseline difference between the concept and main-task loss for first orthogonal concept and main-task vector
    """

    # get the data from data object
    X_train = data_obj.X_train
    X_val = data_obj.X_val
    y_m_val = data_obj.y_m_val
    y_c_val = data_obj.y_c_val
    y_m_train = data_obj.y_m_train
    y_c_train = data_obj.y_c_train
    d = X_val.shape[1]

    
    # get the model
    joint_model = return_joint_main_concept_model(d,  
                                                  loaders, 
                                                  device,
                                                               solver=solver,
                                                               lr=lr,
                                                               weight_decay=weight_decay,
                                                               per_step=per_step,
                                                               tol=tol,
                                                               early_stopping=early_stopping,
                                                               patience=patience,
                                                               epochs=epochs,
                                                               bias=bias,
                                                               model_name = model_name,
                                                               save_best_model=True
                                                               )


    # use training or test data
    if use_training_data_for_testing:
       output = joint_model(X_train)
    else:
        output = joint_model(X_val)
    
    # get the predictions
    y_c_pred_baseline = output['y_c_1']
    y_m_pred_baseline = output['y_m_1']

    # if use_training_data_for_testing, then use the training data for the baseline
    if use_training_data_for_testing:

      
        BCE_concept = torch_calc_BCE(
                        y_c_train, y_c_pred_baseline, device, reduce='none', weighted=eval_balanced) 
        BCE_main = torch_calc_BCE(
                        y_m_train, y_m_pred_baseline, device, reduce='none', weighted = eval_balanced) 
    # else use the validation data
    else:

        
        BCE_concept = torch_calc_BCE(
                                    y_c_val, y_c_pred_baseline, device, reduce='none', weighted=eval_balanced) 
        BCE_main = torch_calc_BCE(
                                    y_m_val, y_m_pred_baseline, device, reduce='none', weighted = eval_balanced) 

    # get the accuracy 
    if use_training_data_for_testing:
        acc_concept = get_acc_pytorch_model(y_c_train, y_c_pred_baseline)
        acc_main = get_acc_pytorch_model(y_m_train, y_m_pred_baseline)
    else:
        acc_concept = get_acc_pytorch_model(y_c_val, y_c_pred_baseline)
        acc_main = get_acc_pytorch_model(y_m_val, y_m_pred_baseline)
    
    acc_diff = acc_concept - acc_main

    print('when calculating baseline, the concept accuracy is: {}'.format(acc_concept))
    print('when calculating baseline, the main-task accuracy is: {}'.format(acc_main))
    print('the difference between the concept and main-task accuracy is: {}'.format(acc_diff))
    print('BCE concept: {}'.format(BCE_concept.mean()))
    print('BCE main: {}'.format(BCE_main.mean()))
    print('BCE diff (concept/main): {}'.format((BCE_concept - BCE_main).mean()))

    # get the difference for each observation
    baseline_diff =  (BCE_concept - BCE_main)

    if eval_balanced:

        groups = data_obj.get_groups(y_c=y_c_val, y_m=y_m_val)
        groups =  torch.Tensor(groups).int().to(device) + 1
        expected_diff = group_weighted_d_mean_estimator(groups, baseline_diff)
        var_1_est, var_2_est, var_3_est, var_4_est = get_different_vars(groups,baseline_diff)

        var_diff = calc_variance_estimator(groups, var_1_est, var_2_est, var_3_est, var_4_est)
    else:

        # get the expected difference
        expected_diff =  baseline_diff.mean().detach()

        # get the variance of the difference
    var_diff =  baseline_diff.var().detach()

    if return_model_results:
        return expected_diff, var_diff, BCE_concept, BCE_main, baseline_diff, joint_model, acc_diff
    else:
        return expected_diff, var_diff
    
def loss_joint_model(y_c, y_m, model_output, device):
    """
    Loss of the joint model for the main-task and concept vector
    """
   
    y_c_pred = model_output['y_c_1']
    y_m_pred = model_output['y_m_1']

    loss_y_c = nn.BCEWithLogitsLoss()(y_c_pred, y_c.float().to(device))
    loss_y_m = nn.BCEWithLogitsLoss()(y_m_pred, y_m.float().to(device))
    loss = loss_y_c + loss_y_m

    return loss

def loss_joint_estimation( y_c, y_m, model_output, d_c, d_m, device, lambda_=1):
    """
    Calculates the loss for the joint estimation problem
    """
    
    # initialize the losses
    loss_y_c = torch.zeros(1).to(device)
    loss_y_m = torch.zeros(1).to(device)
    
    # for the concept
    for i in range(1, d_c+1):
        y_c_pred_i = model_output['y_c_'+str(i)]
        loss_y_c_i = nn.BCEWithLogitsLoss()(y_c_pred_i, y_c.unsqueeze(1).float().to(device))
        loss_y_c = loss_y_c + loss_y_c_i
    
    # for the main task
    for j in range(1, d_m+1):
        y_m_pred_j = model_output['y_m_'+str(j)]
        loss_y_m_j = nn.BCEWithLogitsLoss()(y_m_pred_j, y_m.unsqueeze(1).float().to(device))
        loss_y_m = loss_y_m + loss_y_m_j
    
    # combine losses
    loss = loss_y_c + lambda_ * loss_y_m
    
    return loss


def return_linear_model(d, loaders, device, solver = 'SGD', lr=1, momentum = 0.9, weight_decay = 0.001,  per_step=10, tol = 0.001, early_stopping = True, patience = 5, epochs = 100, bias=False, model_name='toy_model_linear', save_best_model=True):
    """
    Returns a trained linear model
    """

    # create linear model
    model = linear_model(feature_size=d, output_size=1, bias=bias)
    model.apply(init_zero_weights) 
    model = model.to(device)
    
    
    # define optimizer
    if solver=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum = momentum, weight_decay = weight_decay)
    elif solver=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
      
    # train model
    train_linear_model(epochs, 
                       model, 
                              loaders, 
                              'train',
                              'val', 
                              per_step=per_step,
                              optimizer=optimizer, 
                              device=device,
                              early_stopping=early_stopping,
                              orig_patience=patience+1,
                              tol=tol,
                              solver=solver,
                              model_name=model_name,
                              save_best_model=save_best_model)
    
    return model

def return_joint_main_concept_model(d,  loaders, device,  solver = 'SGD', weight_decay=0, lr=1, momentum=0.9, tol=0.0001, patience=5, per_step=10, early_stopping = True, epochs = 100, bias=False, model_name='joint', save_best_model=True):
    """
    Returns a trained joint model for the main-task and concept vector
    """
    

    # create model
    model = joint_main_concept_model(d=d, bias=bias, device=device)
    model.apply(init_zero_weights) 
    model = model.to(device)
    
    # define optimizer
    if solver=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum = momentum, weight_decay=weight_decay)
    elif solver=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
   

    train_joint_model(epochs, 
            model, 
            loaders, 
            'train',
            'val', 
            per_step=per_step,
            optimizer=optimizer, 
            device=device,
            early_stopping=early_stopping,
            orig_patience=patience+1,
            tol=tol,
            solver=solver,
            d_c=1,
            d_m=1,
            model_name=model_name,
            save_best_model=save_best_model
            )
    
    
    return model



def get_t_stat_random_test(y_pred, y, p):
    """
    Get the t-statistic for the difference in BCE loss between the predictions and random predictions (unweighted)
    """
    
    # mean BCE difference, std BCE difference
    BCE_diff = get_BCE_diff_random_classifier(y, y_pred, p)
    mean_BCE_diff_random_y = BCE_diff.mean()
    std_BCE_diff_random_y = BCE_diff.std()
    
    # calculate t-statistic
    n = y_pred.shape[0]
    t_stat_random = mean_BCE_diff_random_y / ((std_BCE_diff_random_y)/np.sqrt(n))

    return t_stat_random

def get_t_stat_compare_concept_main(y_c_pred, y_m_pred, y_c, y_m, device, Delta = 0,  ):
    """
    Get the t-statistic for the difference in BCE loss between the predictions for the concept and main task (unweighted)
    """
    
    # mean BCE difference, std BCE difference
    mean_BCE_diff, std_BCE_diff = calc_BCE_difference_stats(y_c_pred, y_m_pred, y_c, y_m, device)

    # calculate t-statistic
    n = y_c.shape[0]
    t_stat_y_c_y_m_pred = (mean_BCE_diff - Delta) / ((std_BCE_diff)/np.sqrt(n))
   
    
    return t_stat_y_c_y_m_pred



def get_v_adjusted(data_obj, X_train, X_val, v_unit, which_dependent, device, solver, lr,  per_step, tol, early_stopping, patience, bias, epochs, batch_size, model_name):
    """
    For a given v, give a v that is adjusted for the concept or main task by (1) multiplying it with a term (gamma) and (2) different intercept
    """
    
    # get linear projection of X_train and X_val on v
    X_train_proj_v_unit = torch.matmul(X_train, v_unit)
    X_val_proj_v_unit = torch.matmul(X_val, v_unit)
    data_obj.reset_X(X_train_proj_v_unit, X_val_proj_v_unit, batch_size = batch_size)

    if which_dependent =='main':
        loader = data_obj.main_loader
    elif which_dependent == 'concept': 
        loader = data_obj.concept_loader

    # get the linear model for the projection
    scalar_model_v_for_y = return_linear_model(1, 
                                              loader,
                                              device,
                                              solver = solver,
                                              lr=lr,
                                              per_step=per_step,
                                              tol = tol,
                                              early_stopping = early_stopping,
                                              patience = patience,
                                              epochs = epochs,
                                              bias=bias,
                                              model_name = model_name, 
                                              weight_decay=0)  
    scalar_v_for_y = scalar_model_v_for_y.linear_layer.weight.data
    v_for_y = scalar_v_for_y*v_unit
    print('scalar_v_for_y: {}'.format(scalar_v_for_y))

    # reset data
    data_obj.reset_X(X_train, X_val, batch_size = batch_size)

    if bias:
        bias_for_y = scalar_model_v_for_y.linear_layer.bias.data
        print('bias_for_y: {}'.format(bias_for_y))
        return v_for_y, bias_for_y
    else:
        return v_for_y



def train_INLP(data_obj,device,  batch_size = 1, solver = 'SGD', lr=1, weight_decay=0, per_step=10, tol = 0.001, early_stopping = True, patience = 5, epochs=100, bias = False, alpha=0.05, model_base_name='toy_model', joint_decision_rule=False, expected_diff=0, var_diff=0, include_weights=False, train_weights = None, val_weights=None, orthogonality_constraint=False,  use_training_data_for_testing=False):
     """
     Training of the INLP algorithm
     """
    
     # save the initial X_train, X_val here
     X_train_orig = data_obj.X_train.to(device)
     X_val_orig = data_obj.X_val.to(device)

     # save for first iteration
     X_i_train =X_train_orig
     X_i_val = X_val_orig
     
     # save the y_c, and y_m
     y_c_val = data_obj.y_c_val.to(device)
     y_c_train = data_obj.y_c_train.to(device)
     y_m_val = data_obj.y_m_val.to(device)

     # set critical value for stopping
     n_val = y_m_val.shape[0]
     critical_value_random = stats.norm.ppf(1-alpha)

     # create a dictionary to save the in outer loop 
     dict_concept_models = {}

     # set dimensionality
     d = data_obj.X_train.shape[1]
     p = y_c_train.float().mean()
     d_c = 0
     
     for i in range(d):

        # reset the X after projecting out the previous concept vectors
        data_obj.reset_X(X_i_train, X_i_val, batch_size = batch_size, include_weights=include_weights, train_weights=train_weights, val_weights=val_weights)
        print(orthogonality_constraint)


        # create the concept model - standard implementation of INLP
        if not orthogonality_constraint:
            concept_model_i = return_linear_model(d, 
                                              data_obj.concept_loader,
                                              device,
                                              solver = solver,
                                              lr=lr,
                                              per_step=per_step,
                                              tol = tol,
                                              early_stopping = early_stopping,
                                              patience = patience,
                                              epochs = epochs,
                                              bias=bias,
                                              model_name=model_base_name,
                                              weight_decay=weight_decay) 
        # alternative - used for ablation on orthogonality constraint
        else:
           concept_model_i = return_joint_main_concept_model(d,
                                                                            data_obj.concept_main_loader,
                                                                            device, 
                                                                            solver = solver, 
                                                                            lr=lr, 
                                                                            weight_decay=weight_decay,
                                                                            per_step=per_step, 
                                                                            tol = tol,
                                                                            early_stopping = early_stopping,
                                                                            patience = patience, 
                                                                            epochs=epochs,
                                                                            bias = bias, 
                                                                            model_name=model_base_name)
        
        # add the concept model to the dict
        dict_concept_models[i+1] = concept_model_i
        
        # check coefficients
        if orthogonality_constraint:
            v_c = concept_model_i.return_coef(type_vec='concept')
        else:
            v_c = concept_model_i.linear_layer.weight.data
        

        v_c_unit = v_c/torch.linalg.norm(v_c, ord=2)

     
        # the basis at i
        if orthogonality_constraint:
            V_i_C = get_V(dict_concept_models, d=d, type_vec='concept', device=device)
            
        else:
            V_i_C = get_V(dict_concept_models, d=d, type_vec='linear_model',  device=device)

        # get predictions for this vector
        if use_training_data_for_testing:
            y_v_c_y_c_pred = concept_model_i(data_obj.X_train)
        else:
            y_v_c_y_c_pred = concept_model_i(data_obj.X_val)

        if orthogonality_constraint:
            y_v_c_y_c_pred = y_v_c_y_c_pred['y_c_1'].to(device)
        else:
            y_v_c_y_c_pred = y_v_c_y_c_pred.to(device)

        # get the z value for the concept
        if use_training_data_for_testing:
            t_c_r = get_t_stat_random_test(y_v_c_y_c_pred, y_c_train, p=p)
        else:
            t_c_r = get_t_stat_random_test(y_v_c_y_c_pred, y_c_val, p=p)


        if use_training_data_for_testing:
            acc_v_c = get_acc_pytorch_model(y_c_train, y_v_c_y_c_pred)
        else:
            acc_v_c = get_acc_pytorch_model(y_c_val, y_v_c_y_c_pred)
        print('accuracy of concept vector for rafvogel, at iteration {}: {}'.format(i, acc_v_c))
        
        if not t_c_r < -critical_value_random:
            print('Stopped at {} in concept loop; not better than  random'.format(i))
            V_i_C = V_i_C[:, 0:-1]
            d_c = i 
            print('d_c: {}'.format(d_c))
            print(V_i_C.shape)
            break

        # Perform Rafvogtel with our decision rule 
        if joint_decision_rule:
       
            if not orthogonality_constraint:
                v_c_unit = v_c_unit.T
            v_c_for_y_m, intercept_v_c_for_y_m = get_v_adjusted(data_obj,data_obj.X_train, data_obj.X_val, v_c_unit, 'main', device, solver, lr,per_step, tol, early_stopping, patience, bias, epochs, batch_size, model_name=model_base_name+'_v_c_for_y_m')
           
            # get test statistic 
            y_v_c_y_m_pred = torch.matmul(X_val_orig, v_c_for_y_m) + intercept_v_c_for_y_m
            t_c_m_concept= get_t_stat_compare_concept_main(y_v_c_y_c_pred.detach(), y_v_c_y_m_pred.detach(), y_c_val, y_m_val, device=device, Delta=expected_diff)#calc_z_y_c_y_m_pooled(y_v_c_y_c_pred.detach(),y_v_c_y_m_pred.detach(), y_c_val, y_m_val, device, expected_diff=expected_diff, var_baseline_diff=var_diff)

            # test if should quit 
            print('t_c_m rafvogel, at iteration {}: {}'.format(i, t_c_m_concept))
            df = y_c_val.shape[0] - 1
            critical_value_compare = stats.norm.ppf(1-alpha)
            if not t_c_m_concept < -critical_value_compare:
                print('quit procedure due to z_v_c rafvogel, at iteration {}: {}'.format(i, t_c_m_concept))
                V_i_C = V_i_C[:, 0:-1]
                d_c = i
                break

        
        P_i = create_P(V_i_C)
        P_i_orth = torch.eye(d)  - P_i


        X_i_train = torch.matmul(X_train_orig, P_i_orth.detach().to(device))
        X_i_val = torch.matmul(X_i_val,  P_i_orth.detach().to(device))
        d_c = V_i_C.shape[1]


     return V_i_C, d_c

def get_V(dict_models, d, type_vec, device, turn_unit=True, P_orth = None):
    """
    Acquire the basis for the concept or main task vectors
    """
    
    # get the number of models
    k = len(dict_models.keys())
    
    # initialize the matrix
    V = torch.zeros((d, k))
    
    # go over each
    for i in range(k):
        
        # select model i 
        model_i = dict_models[i+1]
        
        # get the coefficient
        if type_vec == 'linear_model':
            v_i = model_i.linear_layer.weight.data.detach().type(torch.float32)
        else:
            v_i =model_i.return_coef(type_vec=type_vec).detach().type(torch.float32).T
        
        # orthogonalize with respect to previous v_i
        if P_orth is not None:
            v_i = torch.matmul(v_i,P_orth.type(torch.float32).to(device))
        
        # if not first vector
        if i > 0:
            
            # if second vector
            if i == 1:
                v_i_min1 = V[:, i-1].unsqueeze(-1).type(torch.float32)
            # if not second vector
            else:
                v_i_min1 = V[:, :i].type(torch.float32)
            
            # define the orthogonal projection matrix based on this
            P_v_i_min1_orth = torch.eye(d) - create_P(v_i_min1)
            
            # define orthogonal vector
            v_i = torch.matmul(v_i, P_v_i_min1_orth.to(device))
            
            # turn unit
            if turn_unit:
                v_i = (v_i/torch.linalg.norm(v_i, ord=2)).type(torch.float32)
        else:
            # for first case, just add to the baiss
            if turn_unit:
                v_i = v_i/torch.linalg.norm(v_i, ord=2).type(torch.float32)
          
        # add to basis
        V[:, i] = v_i.squeeze(-1)
     
    
    return V


def train_JSE(data_obj, device, batch_size = 128, solver = 'SGD', lr=0.01, weight_decay=0,  per_step=1, tol = 0.001, early_stopping = True, patience = 5, epochs=100, alpha = 0.05, Delta = 0,   null_is_concept=False,  model_base_name='toy_model', include_weights = False, train_weights = None, val_weights = None, eval_balanced=True, concept_first=True):
     """
     Run the JSE algorithm 
     """
    
     # save the initial X_train, X_val here
     X_train_orig = data_obj.X_train.to(device)
     X_eval_orig = data_obj.X_val.to(device)
    
     # save for first iteration
     X_i_train =X_train_orig
     X_i_eval = X_eval_orig
 
     # save the y_c, and y_m
     y_c_train = data_obj.y_c_train.to(device)
     y_m_train = data_obj.y_m_train.to(device)

     # save the y_c, and y_m
     y_c_eval = data_obj.y_c_val.to(device)
     y_m_eval = data_obj.y_m_val.to(device)

     # in the case that the evaluation is balanced, then use the balanced weights
     groups = Dataset_obj.get_groups(y_c=y_c_eval, y_m=y_m_eval)
     n_per_group = Dataset_obj.get_n_per_class(torch.Tensor(groups).int(), classes= [0, 1,2,3])
     eval_weights = Dataset_obj.give_balanced_weights(groups, len(groups), n_per_group, normalized=False).unsqueeze(-1)

     # turn the list groups into a tensor
     groups = torch.Tensor(groups).int().to(device) + 1


     # calculate probability for random classifier
     # if use balanced evaluation, then use 0.5
     if eval_balanced:
        p_c = 0.5
        p_m = 0.5
     else:
        p_c = y_c_train.float().mean()
        p_m = y_m_train.float().mean()

     # create a dictionary to save the models in outer loop 
     dict_concept_models = {}

    # get the number of features
     d = X_i_train.shape[1]

     # set critical value for stopping with random test
     n_val = y_c_eval.shape[0] 
     df_t_test = n_val - 1
     critical_value_random = stats.norm.ppf(1-alpha)
     critical_value_compare = stats.norm.ppf(1-alpha)

     
     # loop d times to find the concept vectors
     for i in range(d):
         
         # set the data for the i-th iteration
         X_i_j_train = X_i_train
         X_i_j_val = X_i_eval
         
         # create a dictionary to save the in inner loop         
         dict_main_models = {}
         
         #  loop d times to find the main-task vectors
         for j in range(d):

             # reset the X_train, X_val to the current iteration
             data_obj.reset_X(X_i_j_train,X_i_j_val , batch_size = batch_size, include_weights = include_weights, train_weights = train_weights, val_weights = val_weights, concept_first=concept_first)
             
             # train the model for orthogonal concept and main-task vectors
             joint_main_concept_model_i_j = return_joint_main_concept_model(d,
                                                                            data_obj.concept_main_loader,
                                                                            device, 
                                                                            solver = solver, 
                                                                            lr=lr, 
                                                                            weight_decay=weight_decay,
                                                                            per_step=per_step, 
                                                                            tol = tol,
                                                                            early_stopping = early_stopping,
                                                                            patience = patience, 
                                                                            epochs=epochs,
                                                                            bias = True, 
                                                                            model_name=model_base_name,
                                                                            save_best_model=True)
                                                                            
             # add to models;
             dict_main_models[j+1] = joint_main_concept_model_i_j
         
             # get the V_m at point i, j
             type_vec_m = 'main'
             if  i > 0:
                 V_i_j_M = get_V(dict_main_models, d=d, type_vec=type_vec_m, P_orth = P_i_j_C_orth, turn_unit=True, device=device)
                 V_i_j_M_coef = get_V(dict_main_models, d=d, type_vec=type_vec_m, P_orth = P_i_j_C_orth, turn_unit=False,  device=device)
             else:
                 V_i_j_M = get_V(dict_main_models, d=d, type_vec=type_vec_m, turn_unit=True,  device=device)
                 V_i_j_M_coef = get_V(dict_main_models, d=d, type_vec=type_vec_m, turn_unit=False,  device=device)
                 
             # get the main-task vector
             v_m = V_i_j_M_coef[:, -1].unsqueeze(-1)
             v_m_unit = V_i_j_M[:, -1].unsqueeze(-1)

             print('At step {} of the main loop, {} of the concept loop'.format(j, i))
             
             # adapt main-task vector for the concept labels
             if concept_first:
                to_adjust_for = 'concept'
                v_m_for_y_c, intercept_v_m_for_y_c = get_v_adjusted(data_obj, data_obj.X_train, data_obj.X_val, v_m_unit.to(device), to_adjust_for, device, solver, lr,  per_step, tol, early_stopping, patience, True, epochs, batch_size, model_name=model_base_name+'_v_m_for_y_c')
                intercept_v_m_for_y_m = joint_main_concept_model_i_j.w_m[1].bias.data[0]  # save the intercept for the main-task - W.bias.data[1]//.w_m[1].bias.data[0]
                v_m_for_y_m =  v_m  # get main-task vector for the main-task labels
                

             else:
                to_adjust_for = 'main'
                v_m_for_y_c = v_m
                intercept_v_m_for_y_c = joint_main_concept_model_i_j.w_m[1].bias.data[0]  # save the intercept for the main-task - W.bias.data[1]//.w_m[1].bias.data[0]
                v_m_for_y_m, intercept_v_m_for_y_m = get_v_adjusted(data_obj, data_obj.X_train, data_obj.X_val, v_m_unit.to(device), to_adjust_for, device, solver, lr,  per_step, tol, early_stopping, patience, True, epochs, batch_size, model_name=model_base_name+'_v_m_for_y_m')
             
            
             # get the predictions for the main-task and concept labels by the main-task vector
             y_v_m_y_m_pred = (torch.matmul(data_obj.X_val, v_m_for_y_m.to(device)) + intercept_v_m_for_y_m).to(device)
             y_v_m_y_c_pred = (torch.matmul(data_obj.X_val, v_m_for_y_c.to(device)) + intercept_v_m_for_y_c).to(device)

             if concept_first:
                predictions_for_random_test =  y_v_m_y_m_pred
                y_eval_random_test = y_m_eval
             else:
                predictions_for_random_test =  y_v_m_y_c_pred
                y_eval_random_test = y_c_eval
             
             # save the accuracy
             acc_y_v_m_y_m = get_weighted_acc_pytorch_model(y_m_eval, y_v_m_y_m_pred, y_c_eval) 
             acc_y_v_m_y_c = get_weighted_acc_pytorch_model(y_c_eval, y_v_m_y_c_pred, y_m_eval) 
             print('accuracy of main-task vector for main-task (weighted): {}'.format(acc_y_v_m_y_m.item()))
             print('accuracy of main-task vector for concept-task (weighted): {}'.format(acc_y_v_m_y_c.item()))
             print('at part {} of the main loop, {} of the concept loop'.format(j, i))
             print('concept first: {}'.format(concept_first))
             

             # Test statistic for test 1# in main loop; if not better than random
             if eval_balanced:
                t_m_r = get_weighted_BCE_diff_t_stat_random(y_eval_random_test, predictions_for_random_test, p_m, groups, 0, n_val)
             else:
                t_m_r = get_t_stat_random_test(predictions_for_random_test, y_eval_random_test, p=p_m)

             # Test statistic for test 2# in main loop; if more information about main-task than concept
             if eval_balanced:
                 print('evaluating based on balanced')
                 t_c_m_main = get_weighted_BCE_diff_t_stat_compare(y_m_eval, y_c_eval, y_v_m_y_m_pred,y_v_m_y_c_pred, groups, Delta, n_val,device )
             else:
                 t_c_m_main = get_t_stat_compare_concept_main(y_v_m_y_c_pred,y_v_m_y_m_pred, y_c_eval, y_m_eval, device,  Delta=-Delta)
             print('t_m_r: {}, t_c_m_main: {}'.format(t_m_r, t_c_m_main))
                
            # Break point 1 in main loop; if not better than random
             if not (t_m_r < -critical_value_random):
                d_m = j  
                print('Ended main-task loop at {} in iteration {} of concept loop; not better than random'.format(j, i))  
                break 
            
             # Break point 2 in main loop; if more information about main-task than concept
             if not concept_first:
                print('changing')
                t_c_m_main = -t_c_m_main
                print('t_c_m_main: {}'.format(t_c_m_main))

             if not (t_c_m_main > critical_value_compare):
                d_m = j
                print('Ended main-task loop at {} in iteration {} of concept loop; not more informative of main-task thank concept'.format(j, i))  
                break 
             
             # At this point, can continue with loop  - create projection matrix to deduct main-task subspace
             P_i_j_M = create_P(V_i_j_M.to(device))
             P_i_j_M_orth = torch.eye(d).to(device)  - P_i_j_M
             
             # project from the data
             X_i_j_train = torch.matmul(X_i_train, P_i_j_M_orth.detach().to(device)) # check if not X_i_j
             X_i_j_val = torch.matmul(X_i_eval, P_i_j_M_orth.detach().to(device))

         # if d_m not defined
         try:
            d_m
         except NameError:
            d_m = j
         
         # Get last model with valid main-task vector
         if d_m ==0:
            last_joint_model = dict_main_models[1]
            P_i_j_M_orth = torch.eye(d)
         else:
            last_joint_model = dict_main_models[d_m] 
         
         # add this model to dict used for concept space
         dict_concept_models[i+1] = last_joint_model
        
         # get the concept basis
         try:
            V_i_j_C = get_V(dict_concept_models, d=d, type_vec= 'concept', P_orth = P_i_j_M_orth, turn_unit = True ,  device=device)
            V_i_j_C_coef = get_V(dict_concept_models, d=d, type_vec= 'concept', P_orth = P_i_j_M_orth, turn_unit = False ,  device=device)
         except NameError:
            V_i_j_C = get_V(dict_concept_models, d=d, type_vec= 'concept', turn_unit = True,  device=device)
            V_i_j_C_coef = get_V(dict_concept_models, d=d, type_vec= 'concept', turn_unit = False ,  device=device)

         # get latest concept vetor
         v_c = V_i_j_C_coef[:, -1].unsqueeze(-1)
         v_c_unit = V_i_j_C[:, -1].unsqueeze(-1)

         # adapt main-task vector for the concept labels
         if concept_first:
                to_adjust_for = 'main'
                v_c_for_y_c = v_c
                intercept_v_c_for_y_c = last_joint_model.w_c.bias.data[0] #  # save the intercept for the main-task - W.bias.data[0]//.w_c.bias.data[0]
                v_c_for_y_m, intercept_v_c_for_y_m = get_v_adjusted(data_obj, X_i_train, X_i_eval, v_c_unit.to(device), to_adjust_for, device, solver, lr, per_step, tol, early_stopping, patience, True, epochs, batch_size, model_name=model_base_name+'_v_c_for_y_m')

         else:
                to_adjust_for = 'concept'
                v_c_for_y_c, intercept_v_c_for_y_c = get_v_adjusted(data_obj, X_i_train, X_i_eval, v_m_unit.to(device), to_adjust_for, device, solver, lr,  per_step, tol, early_stopping, patience, True, epochs, batch_size, model_name=model_base_name+'_v_m_for_y_c')
                v_c_for_y_m = v_c
                intercept_v_c_for_y_m = last_joint_model.w_c.bias.data[0] #  # save the intercept for the main-task - W.bias.data[0]//.w_c.bias.data[0]
                
                
         # get the predictions
         y_v_c_y_c_pred = torch.matmul(X_i_eval, v_c_for_y_c.to(device)) + intercept_v_c_for_y_c
         y_v_c_y_m_pred = torch.matmul(X_i_eval, v_c_for_y_m.to(device)) + intercept_v_c_for_y_m

         if concept_first:
            predictions_for_random_test =  y_v_c_y_c_pred
            y_eval_random_test = y_c_eval
         else:
            predictions_for_random_test =  y_v_c_y_m_pred
            y_eval_random_test = y_m_eval
             

         # save the accuracy
         acc_y_v_c_y_c = get_weighted_acc_pytorch_model(y_c_eval, y_v_c_y_c_pred, y_m_eval)
         acc_y_v_c_y_m = get_weighted_acc_pytorch_model(y_m_eval, y_v_c_y_m_pred, y_c_eval)
         print('accuracy of concept vector for concept/main (weighted) at step {}: {}/{}'.format(i, acc_y_v_c_y_c, acc_y_v_c_y_m))
       
         # Test statistic for test 1# in concept loop; if not better than random
         if eval_balanced:
            t_c_r = get_weighted_BCE_diff_t_stat_random(y_eval_random_test, y_v_c_y_c_pred, p_c, groups, 0, n_val)
         else:
            t_c_r = get_t_stat_random_test(y_v_c_y_c_pred, y_eval_random_test, p=p_c)
            
         # Test statistic for test 2# in concept loop; if more information about concept than main-task
         if eval_balanced:
            t_c_m_concept = get_weighted_BCE_diff_t_stat_compare(y_m_eval, y_c_eval, y_v_c_y_m_pred,y_v_c_y_c_pred, groups, Delta, n_val, device )
         else:
            t_c_m_concept = get_t_stat_compare_concept_main(y_v_c_y_c_pred,y_v_c_y_m_pred, y_c_eval, y_m_eval, device, Delta=Delta)

         print('t_c_r: {}, t_c_m_concept: {}'.format(t_c_r, t_c_m_concept))
            

         # Test 1# in concept loop; if not better than random
         if not t_c_r < -critical_value_random:
            V_i_j_C = V_i_j_C[:, 0:-1]
            d_c = i
            print('Stopped at {} in concept loop; not better than  random'.format(i))
            break 


         # Test 2# in concept loop; if more information about concept than main-task
         if not concept_first:
            t_c_m_concept = -t_c_m_concept
               
         if not (t_c_m_concept < -critical_value_compare): 
            print('Stopped at {} in concept loop; not more informative at concept than main-task'.format(i))


            if null_is_concept:
               print('Null is concept; adding here, then stopping')
               d_c = i  
            else:
                V_i_j_C = V_i_j_C[:, 0:-1]
                d_c = i
                break 
                
                
         # projection matrices
         P_i_j_C = create_P(V_i_j_C)
         P_i_j_C_orth = torch.eye(d).to(device)  - P_i_j_C.to(device)

         # change the X
         X_i_train = torch.matmul(X_train_orig, P_i_j_C_orth.detach().to(device))
         X_i_eval = torch.matmul(X_eval_orig, P_i_j_C_orth.detach().to(device))
         
         # set d_c
         d_c = i+1
         
     # get the concept basis
     V_c = V_i_j_C

     # save the last set of main-task vectors
     V_m = V_i_j_M[:, 0:-1]
  
     return V_c, V_m, d_c, d_m
    

def train_joint_model(num_epochs, model, loader_dict, train_loader, val_loader, device, optimizer,d_c =1, d_m=1, lambda_=1, per_step=10, early_stopping= True, orig_patience=6, tol = 0.001, solver='lbfgs', min_loss=True, save_best_model=True, model_name='joint_'):
    
    # total steps to take
    total_step_train = len(loader_dict[train_loader])
    total_step_val = len(loader_dict[val_loader])

    # placeholders to track loss
    best_loss = math.inf
    patience = orig_patience 

    # go over each epoch
    for epoch in range(num_epochs):

      # Each epoch has a training and validation phase
      for phase in [train_loader, val_loader]:
        total_samples = 0
        total_correct_concept_dict = dict.fromkeys(list(range(1, d_c +1)), 0.0)
        total_correct_main_dict = dict.fromkeys(list(range(1, d_m +1)), 0.0)
        total_loss_val = 0
        total_loss_train = 0

        if phase == train_loader:
          model.train()  # Set model to training mode
        else:
          model.eval()   # Set model to evaluate mode

        # go over batches
        for i, (x, y_c, y_m) in enumerate(loader_dict[phase]):

          # input and output batch
          b_x =Variable(x).to(device)
          b_y_m =Variable(y_m).type(torch.LongTensor).to(device)
          b_y_c =Variable(y_c).type(torch.LongTensor).to(device)


          # if training, set step 
          if phase == train_loader:
            
            # if lbfgs solver, use this version
            if solver=='lbfgs':
              
                def closure():
                  # clear zero gradients
                  optimizer.zero_grad()
                  
                  # get output and calc loss
                  model_output = model(b_x.float().to(device))
              
                  loss = loss_joint_estimation( y_c, y_m, model_output, d_c, d_m, device, lambda_=1)
                  
                  
                  # calculate grad
                  loss.backward()
                  
                  return loss
                
                # take step 
                optimizer.step(closure)
                
            else:
                
                # get output and calc loss
                model_output = model(b_x.float().to(device))

                loss =  loss_joint_estimation( y_c, y_m, model_output, d_c, d_m, device, lambda_=1)
          
                # calc grad
                loss.backward()
           
                # take step
                optimizer.step()
                
                # add to loss
                total_loss_train += loss.detach().item()
                    
                # clear gradients for this step   
                optimizer.zero_grad()  
                
              
            
            ## This part is not for optimization but to check performance during training
            if solver =='lbfgs':
                
               # get output and calc loss
               model_output = model(b_x.float().to(device))
           
               loss =  loss_joint_estimation( y_c, y_m, model_output, d_c, d_m, device, lambda_=1)
               
                
            total_loss_train += loss.item()
              
          else:
          ## This part is not for optimization but to check performance during training
            if solver =='lbfgs':
                
              # get output and calc loss
              model_output = model(b_x.float().to(device))
          
              loss =  loss_joint_estimation( y_c, y_m, model_output, d_c, d_m, device, lambda_=1)
              
              
            total_loss_val += loss.item()  
      
          # the predicted classes
          # get output and calc loss
          model_output = model(b_x.float().to(device))

          for j in range(1, d_c+1):
              y_c_j_pred = model_output['y_c_'+str(j)]
              classes_concept_j = torch.round(torch.sigmoid(y_c_j_pred))
              correct_concept_j = (classes_concept_j == b_y_c.unsqueeze(1).to(device)).sum()
              total_correct_concept_dict[j] += correct_concept_j

              #print('the accuracy for this batch (concept) is: {}'.format(correct_concept_j/b_x.shape[0]))
         
          for j in range(1, d_m+1):
              y_m_j_pred = model_output['y_m_'+str(j)]
              classes_main_j = torch.round(torch.sigmoid(y_m_j_pred))
              correct_main_j = (classes_main_j == b_y_m.unsqueeze(1).to(device)).sum()
              total_correct_main_dict[j] += correct_main_j

              #print('the accuracy for this batch (main) is: {}'.format(correct_main_j/b_x.shape[0]))
              
              
        
          
          # total samples 
          total_samples += b_x.shape[0]



          # each .. batches, show progress
          if (i+1) % per_step == 0:

            if phase == train_loader:
              total_step = total_step_train
            else:
              total_step = total_step_val 
            print ('Phase : {}, Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(phase,
                                  epoch + 1, # current epoch
                                  num_epochs, # total epochs
                                  i + 1, # current steps
                                  total_step, # total steps
                                  loss.item(), # loss for this batch
                            )
                          )
            
            for i in range(1, d_c+1):
                print('accuracy concept predictor {} : {:.4f}'.format(i, total_correct_concept_dict[i]/total_samples))
            
            for i in range(1, d_m+1):
                print('accuracy main predictor {} : {:.4f}'.format(i, total_correct_main_dict[i]/total_samples))
            
            
      
      if early_stopping:
          
        avg_loss_val = total_loss_val/total_step_val

        if avg_loss_val > (best_loss - tol):
          patience = patience - 1
          if patience ==0:
              print('Early stopping at epoch{}: current loss {}  > {}, best loss.'.format(epoch, avg_loss_val, best_loss))
              if save_best_model:
                 print('loading best param')
                 model.load_state_dict(torch.load(model_name+'_current_best_model_parameters.pt'))
              break
          else:
              print('Not improving at epoch {}, current loss {} > {}, patience is now {}'.format(epoch + 1, avg_loss_val, best_loss, patience))
        else:
          print('Improving: current loss {} < {}, loss last epoch '.format(avg_loss_val, best_loss))
          best_loss = avg_loss_val
          patience = orig_patience
          
          if save_best_model:
             print('save best param')
             torch.save(model.state_dict(), model_name+'_current_best_model_parameters.pt') # official recommended
             

      else:
          avg_loss_train = total_loss_train/total_step_train

          if avg_loss_train > (best_loss - tol):
            patience = patience - 1
            if patience ==0:
                print('Converged based on train loss at epoch{}: current loss {}  > {}, best loss. Loading best parameters'.format(epoch, avg_loss_train, best_loss))
                if save_best_model:
                   print('loading best param')
                   model.load_state_dict(torch.load(model_name+'_current_best_model_parameters.pt'))
                break
            else:
                print('Not improving at epoch {}, current loss + tolerance {} > {}, patience is now {}'.format(epoch + 1, avg_loss_train + tol, best_loss, patience))
          else:
            print('Improving: current loss {} < {}, loss last epoch '.format(avg_loss_train, best_loss))
            best_loss = avg_loss_train
            patience = orig_patience
            
            
            if save_best_model:
               print('save best param')
               torch.save(model.state_dict(), model_name+'_current_best_model_parameters.pt') # official recommended


def train_linear_model(num_epochs, model, loader_dict, train_loader, val_loader, device, optimizer, per_step=10, early_stopping= True, orig_patience=6, tol = 0.001, solver='lbfgs', min_loss=True, save_best_model=True, model_name='toy_model_linear'):
    
    # total steps to take
    total_step_train = len(loader_dict[train_loader])
    total_step_val = len(loader_dict[val_loader])

    # placeholders to track loss
    best_loss = math.inf
    patience = orig_patience 

    # go over each epoch
    for epoch in range(num_epochs):
     


      # Each epoch has a training and validation phase
      for phase in [train_loader, val_loader]:

        total_samples = 0
        total_correct = 0

        if phase == train_loader:
           total_loss_train = 0
        else:
           total_loss_val = 0
       

        if phase == train_loader:
          model.train()  # Set model to training mode
        else:
          model.eval()   # Set model to evaluate mode

        # go over batches
        for i, (x, y_m) in enumerate(loader_dict[phase]):

          # input and output batch
          b_x =Variable(x)
          b_y_m =Variable(y_m).type(torch.LongTensor)           
         
          
          # if training, set step 
          if phase == train_loader:
            
            # if lbfgs solver, use this version
            if solver=='lbfgs':
              
                def closure():
                  # clear zero gradients
                  optimizer.zero_grad()
                  
                  # get output and calc loss
                  y_m_pred = model(b_x.float().to(device))
              
                  # calculate the joint loss
                  loss = nn.BCEWithLogitsLoss()(y_m_pred, y_m.unsqueeze(1).float().to(device))
                  
                  # turn minimizer to maximizer
                  if not min_loss:
                      
                      ## loss of a random classifier
                      prob_y = b_y_m.float().mean()
                      #y_random = torch.Tensor([prob_y]*len(b_y_m))
                      #loss_random = torch.nn.BCELoss(reduce='mean')(y_random, y_m.float())
                      
                      loss = -loss#torch.abs(loss - loss_random)
                  
                      
                  # calculate grad
                  loss.backward()
                  
                  if not min_loss:
                      model.w_tilde.weight.grad.data = -model.w_tilde.weight.grad.data
                  
                 
                      
                  
                  return loss
                
                # take step 
                optimizer.step(closure)
                
            else:
                
                # get output
                y_m_pred =  model(b_x.float().to(device))

                
                # calc the loss
                loss = nn.BCEWithLogitsLoss()(y_m_pred, y_m.unsqueeze(1).float().to(device))
                
                # turn minimizer to maximizer
                if not min_loss:
                    loss = -loss
                    

                # calc grad
                loss.backward()
                
                if not min_loss:
                    model.w_tilde.weight.grad.data = -model.w_tilde.weight.grad.data
                    
           
                # take step
                optimizer.step()
                
              
                
                # add to loss
                total_loss_train += loss.item()
                    
                # clear gradients for this step   
                optimizer.zero_grad()  
    
            
            
            ## This part is not for optimization but to check performance during training
            if solver =='lbfgs':
                # get output
                y_m_pred = model(b_x.float().to(device))
                
                # calculate the joint loss
                loss = nn.BCEWithLogitsLoss()(y_m_pred, y_m.unsqueeze(1).float().to(device))
                
            total_loss_train += loss.item()
              
          else:
          ## This part is not for optimization but to check performance during training
            if solver =='lbfgs':
              # get output
              y_m_pred = model(b_x.float().to(device))
                  
              # calculate the joint loss
              loss = nn.BCEWithLogitsLoss()(y_m_pred, y_m.unsqueeze(1).float().to(device))
                  
            total_loss_val += loss.item()  
      
          # the predicted classes
          y_m_pred = model(b_x.float().to(device))
          classes =  torch.round(torch.sigmoid(y_m_pred))
          
          # correct main, concept
          correct = (classes == b_y_m.unsqueeze(1).to(device)).sum()
          
          # running amount correct & samples considered
          total_correct += correct
          
          # total samples 
          total_samples += len(b_x) 
          

          # each .. batches, show progress
          if (i+1) % per_step == 0:

            if phase == train_loader:
              total_step = total_step_train
            else:
              total_step = total_step_val 
            print ('Phase : {}, Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, accuracy: {:.4f}' 
                          .format(phase,
                                  epoch + 1, # current epoch
                                  num_epochs, # total epochs
                                  i + 1, # current steps
                                  total_step, # total steps
                                  loss.item(), # loss for this batch
                                  total_correct/total_samples,
                            )
                          )
            
      
      if early_stopping:
          
        avg_loss_val = total_loss_val/total_step_val

        if avg_loss_val > (best_loss - tol):
          patience = patience - 1
          if patience ==0:
              print('Early stopping at epoch{}: current loss {}  > {}, best loss.'.format(epoch, avg_loss_val, best_loss))
              if save_best_model:
                 print('loading best param')
                 model.load_state_dict(torch.load(model_name + '_current_best_model_parameters.pt'))
              break
          else:
              print('Not improving at epoch {}, current loss {} > {}, patience is now {}'.format(epoch + 1, avg_loss_val, best_loss, patience))
        else:
          print('Improving: current loss {} < {}, loss last epoch '.format(avg_loss_val, best_loss))
          best_loss = avg_loss_val
          patience = orig_patience
          
          if save_best_model:
             print('save best param')
             torch.save(model.state_dict(), model_name + '_current_best_model_parameters.pt') # official recommended
             

      else:
          avg_loss_train = total_loss_train/total_step_train

          if avg_loss_train > (best_loss - tol):
            patience = patience - 1
            if patience ==0:
                print('Converged based on train loss at epoch{}: current loss {}  > {}, best loss. Loading best parameters'.format(epoch, avg_loss_train, best_loss))
                if save_best_model:
                   print('loading best param')
                   model.load_state_dict(torch.load(model_name + '_current_best_model_parameters.pt'))
                break
            else:
                print('Not improving at epoch {}, current loss + tolerance {} > {}, patience is now {}'.format(epoch + 1, avg_loss_train + tol, best_loss, patience))
          else:
            print('Improving: current loss {} < {}, loss last epoch '.format(avg_loss_train, best_loss))
            best_loss = avg_loss_train
            patience = orig_patience
            
            
            if save_best_model:
               print('save best param')
               torch.save(model.state_dict(), model_name + '_current_best_model_parameters.pt') # official recommended
               

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



def train_resnet50(num_epochs, model, W, loader_dict, train_loader, val_loader, device, optimizer,  per_step=10, early_stopping= False, orig_patience=5, tol = 0.0001, use_scheduler=False, scheduler=None, save_best_model=True, model_name='toy_model_linear', loss_fn =   nn.BCEWithLogitsLoss(), multiclass=False, adversarial = False, adv_W=None, adv_lambda=1 ,baseline_acc_concept=0.7  ):
   
    total_step_train = len(loader_dict[train_loader])
    total_step_val = len(loader_dict[val_loader])

   # scheduler for training process
    if scheduler =='linear':
        scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=total_step_train*num_epochs)

    # placeholders to track loss
    best_loss = math.inf
    patience = orig_patience


    for epoch in range(num_epochs):
       
      total_samples_train = 0
      total_samples_val = 0
      total_correct_train = 0
      total_correct_val = 0
      total_loss_val = 0
      total_loss_train = 0

      if adversarial:
         total_correct_train_adv = 0
         total_correct_val_adv = 0

      # Each epoch has a training and validation phase
      for phase in [train_loader, val_loader]:
        if phase == train_loader:
          model.train()  # Set model to training mode
        else:
          model.eval()   # Set model to evaluate mode

        # go over batches
        for i, batch in enumerate(loader_dict[phase]):
           
           b_x = batch[0].to(device)
           b_y_m = batch[1].to(device).unsqueeze(1).float()

           if adversarial:
              b_y_c = batch[2].to(device).unsqueeze(1).float()


           # if training, set step 
           if phase == train_loader:
             
                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because 
                # accumulating the gradients is "convenient while training RNNs". 
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
               
                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                classifier_input  = model(b_x)

                logits = W(classifier_input)

                if adversarial:
                   logits_adv = adv_W(classifier_input)

                # Compute loss and accumulate the loss values
                
                loss = loss_fn(logits, b_y_m)
                
                # Adding at this point; total loss train based on main-task loss
                total_loss_train += loss.item()

                if adversarial:
                     loss_adv = loss_fn(logits_adv, b_y_c)
                     loss += adv_lambda*loss_adv
                    



                total_samples_train += len(b_y_m)

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # get the predictions
                pred = torch.round(torch.sigmoid(logits))
                correct = pred.eq(b_y_m).sum()
                total_correct_train += correct.item()
                acc = correct.float()/b_y_m.shape[0]
                print('Accuray at batch {} is {}'.format(i, acc))


                if adversarial:
                       pred_adv = torch.round(torch.sigmoid(logits_adv))
                       print('pred adv is {}'.format(pred_adv))

                       correct_adv = pred_adv.eq(b_y_c).sum()
                       acc_adv = correct_adv.float()/b_y_c.shape[0]
                       print('Accuray at batch {} of adversary is {}'.format(i, acc_adv))

                       # add tot total correct
                       total_correct_train_adv += correct_adv.item()

                       print('p(y_c = 1) for batch {} is {}'.format(i, b_y_c.float().mean()))
                       print('p(y_m = 1) for batch {} is {}'.format(i, b_y_m.float().mean()))

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                if use_scheduler:
                    # Update the learning rate.
                    scheduler.step()

                 # clear gradients for this step   
                optimizer.zero_grad()  
                
                # clear the memory
                del b_y_m
                del b_x
                


           else:
                model.eval()
                with torch.no_grad():
                   
                    classifier_input  = model(b_x)
                    logits = W(classifier_input)

                        

                    loss = loss_fn(logits, b_y_m)

                    # Adding at this point; total loss train based on main-task loss
                    total_loss_val += loss.item()

                    if adversarial:
                        adv_output = adv_W(classifier_input)
                        loss += loss_fn(adv_output, b_y_c) * adv_lambda



            

                    # Accumulate the training loss over all of the batches so that we can
                    # calculate the average loss at the end. `loss` is a Tensor containing a
                    # single value; the `.item()` function just returns the Python value 
                    # from the tensor.
                    total_samples_val += len(b_y_m) 

                    if adversarial:
                       pred_adv = torch.round(torch.sigmoid(adv_output))
                       correct_adv = pred_adv.eq(b_y_c.float()).sum()
                       acc_adv = correct_adv.float()/b_y_c.shape[0]
                       print('Accuray at batch {} of adversary is {}'.format(i, acc_adv))

                       total_correct_val_adv += correct_adv.item()
            

                    # calculate the accuracy 
                    pred = torch.round(torch.sigmoid(logits))
                    correct = pred.eq(b_y_m).sum()
                    acc = correct.float() / b_y_m.shape[0]
                    print('Accuracy at batch {:} is {:}'.format(i, acc))
                    total_correct_val += correct.item()


                   
               
               
          

            # each .. batches, show progress
           if (i+1) % per_step == 0:

                if phase == train_loader:
                    total_step = total_step_train
                    total_samples = total_samples_train
                    correct_step = total_correct_train
                else:
                    total_step = total_step_val 
                    total_samples = total_samples_val
                    correct_step = total_correct_val

                message = 'Phase : {}, Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, accuracy (main): {:.4f}'.format(phase,
                                    epoch + 1, # current epoch
                                    num_epochs, # total epochs
                                    i + 1, # current steps
                                    total_step, # total steps
                                    loss.item(), # loss for this batch
                                    correct_step/total_samples,
                                )
                
              
                if adversarial:
                    if phase == train_loader:
                       correct_step_adv = total_correct_train_adv
                    else:
                       correct_step_adv = total_correct_val_adv
                    message += ', accuracy (concept): {:.4f}, difference with random classifier: {:.4f}'.format(correct_step_adv/total_samples, correct_step_adv/total_samples - baseline_acc_concept )
                print(message)
            
            
      
      if early_stopping:
          
        avg_loss_val = total_loss_val/total_step_val

        if avg_loss_val > (best_loss - tol):
          patience = patience - 1
          if patience ==0:
              print('Early stopping at epoch{}: current loss {}  > {}, best loss.'.format(epoch, avg_loss_val, best_loss))
              if save_best_model:
                 print('loading best param')
                 model.load_state_dict(torch.load(model_name + '_current_best_model_parameters.pt'))
              break
          else:
              print('Not improving at epoch {}, current loss {} > {}, patience is now {}'.format(epoch + 1, avg_loss_val, best_loss, patience))
        else:
          print('Improving: current loss {} < {}, loss last epoch '.format(avg_loss_val, best_loss))
          best_loss = avg_loss_val
          patience = orig_patience
          
          if save_best_model:
             print('save best param')
             torch.save(model.state_dict(), model_name + '_current_best_model_parameters.pt') # official recommended
             
      else:
       
        avg_loss_train = total_loss_train/total_step_train

        if avg_loss_train > (best_loss - tol):
            print('Not improving at epoch {}, current loss {} > {}'.format(epoch + 1, avg_loss_train, best_loss))
        else:
            print('Improving: current loss {} < {}, loss last epoch '.format(avg_loss_train, best_loss))
            best_loss = avg_loss_train

            if save_best_model:
                print('save best param')
                torch.save(model.state_dict(), model_name + '_current_best_model_parameters.pt') # official recommended
        


    if save_best_model:
       print('At end, load best param')
       model.load_state_dict(torch.load(model_name + '_current_best_model_parameters.pt'))
     

                                


      
       

def train_BERT(num_epochs, finetune_mode, model, W, loader_dict, train_loader, val_loader, device, optimizer, dropout=None, per_step=10, early_stopping= False, orig_patience=5, tol = 0.0001, use_scheduler=False, scheduler=None, save_best_model=True, model_name='toy_model_linear', loss_fn =   nn.BCEWithLogitsLoss(), multiclass=False, adversarial = False, adv_W=None, adv_lambda=1.0   ):
    
    # total steps to take
    total_step_train = len(loader_dict[train_loader])
    total_step_val = len(loader_dict[val_loader])

    # scheduler for training process
    if scheduler =='linear':
       scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=total_step_train*num_epochs)

    # placeholders to track loss
    best_loss = math.inf
    patience = orig_patience 

    # go over each epoch
    for epoch in range(num_epochs):
      
      t0 = time.time()
      total_samples_train = 0
      total_samples_val = 0
      total_correct_train = 0
      total_correct_val = 0
      total_loss_val = 0
      total_loss_train = 0

      if adversarial:
         total_correct_train_adv = 0
         total_correct_val_adv = 0


      # Each epoch has a training and validation phase
      for phase in [train_loader, val_loader]:
        if phase == train_loader:
          model.train()  # Set model to training mode
        else:
          model.eval()   # Set model to evaluate mode

        # go over batches
        for i, batch in enumerate(loader_dict[phase]):

            b_input_ids = batch[0].to(device)
            b_token_type_ids = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)
            print('p(y_m = 1) for batch {} is {}'.format(i, b_labels.float().mean()))

            if adversarial:
               b_concept = batch[4].to(device)
               print('p(y_c = 1) for batch {} is {}'.format(i, b_concept.float().mean()))
          
          
         
          
            # if training, set step 
            if phase == train_loader:
             
                
                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                output  = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask
                                )
                
                if finetune_mode == 'pooler':
                   classifier_input = output['pooler_output']
                elif finetune_mode == 'CLS':
                   classifier_input = output['last_hidden_state'][:,0,:]
                
                if dropout is not None:
                   print('use the dropout')
                   classifier_input = dropout(classifier_input)
                   
                logits = W(classifier_input)

                
                
                if not multiclass:
                   b_labels = b_labels.unsqueeze(1)
                loss = loss_fn(logits, b_labels.float())

                # Add to total loss train for main task
                total_loss_train += loss.detach().item()

                if adversarial:
                   adv_output = adv_W(classifier_input)
                   b_concept = b_concept.unsqueeze(1)
                   loss_adv = loss_fn(adv_output, b_concept.float())

                  
                   
                   loss += (loss_adv * adv_lambda)

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                total_samples_train += len(b_labels) 

                

                # calculate the accuracy 

                if not multiclass:
                    pred = torch.round(torch.sigmoid(logits))
                    correct = pred.eq(b_labels.float()).sum()


                    if adversarial:
                       pred_adv = torch.round(torch.sigmoid(adv_output))
                       print("pred adv is {}".format(pred_adv[0:4]))
                       print('actual concept is {}'.format(b_concept[0:4]))
                       correct_adv = pred_adv.eq(b_concept.float()).sum()
                       acc_adv = correct_adv.float()/b_concept.shape[0]
                       print('Accuray at batch {} of adversary is {}'.format(i, acc_adv))

                       # add tot total correct
                       total_correct_train_adv += correct_adv.item()
                else:
                    pred = torch.max(logits, dim=1).indices
                    correct_labels = torch.max(b_labels, dim=1).indices
              

                    correct = pred.eq(correct_labels).sum()
                acc = correct.float() / b_labels.shape[0]
                print('Accuracy at batch {:} is {:}'.format(i, acc))
                print('Probability of y_m = 1 is {}'.format(b_labels.float().mean()))

                total_correct_train += correct.item()
                

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.

                optimizer.step()

                if use_scheduler:
                    # Update the learning rate.
                    scheduler.step()
                

                # clear gradients for this step   
                optimizer.zero_grad()  
                
                # clear the memory
                del b_input_ids
                del b_token_type_ids
                del b_input_mask
                del b_labels
                del output


        
            
            
            else:
                model.eval()
                with torch.no_grad():
                   
                    output  = model(b_input_ids, 
                                    token_type_ids=b_token_type_ids, 
                                    attention_mask=b_input_mask
                                )
                    if finetune_mode == 'pooler':
                        classifier_input = output['pooler_output']
                    elif finetune_mode == 'CLS':
                        classifier_input = output['last_hidden_state'][:,0,:]
                    
                    logits = W(classifier_input)

                        

                    loss = loss_fn(logits, b_labels.unsqueeze(1).float())

                    # Add to total loss val for main task
                    total_loss_val += loss.item()

                    if adversarial:
                        b_concept = b_concept.unsqueeze(1)
                        adv_output = adv_W(classifier_input)
                        loss += (loss_fn(adv_output, b_concept.float()) * adv_lambda)



            

                    # Accumulate the training loss over all of the batches so that we can
                    # calculate the average loss at the end. `loss` is a Tensor containing a
                    # single value; the `.item()` function just returns the Python value 
                    # from the tensor.
                    total_samples_val += len(b_labels) 

                    if adversarial:
                       pred_adv = torch.round(torch.sigmoid(adv_output))
                       correct_adv = pred_adv.eq(b_concept.float()).sum()
                       acc_adv = correct_adv.float()/b_concept.shape[0]
                       print('Accuray at batch {} of adversary is {}'.format(i, acc_adv))

                       total_correct_val_adv += correct_adv.item()
            

                    # calculate the accuracy 
                    pred = torch.round(torch.sigmoid(logits))
                    correct = pred.eq(b_labels.unsqueeze(1).float()).sum()
                    acc = correct.float() / b_labels.shape[0]
                    print('Accuracy at batch {:} is {:}'.format(i, acc))
                    total_correct_val += correct.item()


                   
               
               
          

            # each .. batches, show progress
            if (i+1) % per_step == 0:


                if phase == train_loader:
                    total_step = total_step_train
                    total_samples = total_samples_train
                    correct_step = total_correct_train
                else:
                    total_step = total_step_val 
                    total_samples = total_samples_val
                    correct_step = total_correct_val

                message = 'Phase : {}, Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, accuracy (main): {:.4f}'.format(phase,
                                    epoch + 1, # current epoch
                                    num_epochs, # total epochs
                                    i + 1, # current steps
                                    total_step, # total steps
                                    loss.item(), # loss for this batch
                                    correct_step/total_samples,
                                )
                
              
                if adversarial:
                    if phase == train_loader:
                       correct_step_adv = total_correct_train_adv
                    else:
                       correct_step_adv = total_correct_val_adv
                    message += ', accuracy (concept): {:.4f}'.format(correct_step_adv/total_samples)
                print(message)
            
            
      
      if early_stopping:
          
        avg_loss_val = total_loss_val/total_step_val

        if avg_loss_val > (best_loss - tol):
          patience = patience - 1
          if patience ==0:
              print('Early stopping at epoch{}: current loss {}  > {}, best loss.'.format(epoch, avg_loss_val, best_loss))
              if save_best_model:
                 print('loading best param')
                 model.load_state_dict(torch.load(model_name + '_current_best_model_parameters.pt'))
              break
          else:
              print('Not improving at epoch {}, current loss {} > {}, patience is now {}'.format(epoch + 1, avg_loss_val, best_loss, patience))
        else:
          print('Improving: current loss {} < {}, loss last epoch '.format(avg_loss_val, best_loss))
          best_loss = avg_loss_val
          patience = orig_patience
          
          if save_best_model:
             print('save best param')
             torch.save(model.state_dict(), model_name + '_current_best_model_parameters.pt') # official recommended
      else:
       
       avg_loss_train = total_loss_train/total_step_train

       if avg_loss_train > (best_loss - tol):
           print('Not improving at epoch {}, current loss {} > {}'.format(epoch + 1, avg_loss_train, best_loss))
       else:
           print('Improving: current loss {} < {}, loss last epoch '.format(avg_loss_train, best_loss))
           best_loss = avg_loss_train

           if save_best_model:
               print('save best param')
               torch.save(model.state_dict(), model_name + '_current_best_model_parameters.pt') # official recommended
    


    if save_best_model:
       print('At end, load best param')
       model.load_state_dict(torch.load(model_name + '_current_best_model_parameters.pt'))
             

      