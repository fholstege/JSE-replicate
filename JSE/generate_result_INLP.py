


import numpy as np
import pandas as pd
import torch


from JSE.data import *
from JSE.settings import data_info, optimizer_info
from JSE.models import *
from JSE.training import *

import argparse
import os
import sys


def main(dataset, dataset_setting, spurious_ratio, demean, pca, k_components, alpha,  batch_size, solver, lr,weight_decay,  early_stopping, epochs,balanced_training_concept, balanced_training_main, rafvogel_with_joint, orthogonality_constraint,  per_step, device_type, save_results, seed, folder, baseline_adjust=False, use_standard_ERM_settings_after=True):

    # set the device
    device = torch.device(device_type)

    # set the settings for dataset
    dataset_settings = data_info[dataset][dataset_setting]
    optimizer_settings = optimizer_info['All']

    results_dict = { 'method': 'INLP', 'parameters': {'spurious_ratio': spurious_ratio, 'demean': demean, 'pca': pca, 'k_components': k_components, 'alpha': alpha,  'batch_size': batch_size, 'solver': solver, 'lr': lr, 'weight_decay': weight_decay, 'early_stopping': early_stopping, 'epochs': epochs, 'balanced_training_concept': balanced_training_concept, 'balanced_training_main': balanced_training_main, 'per_step': per_step, 'device_type': device_type, 'seed': seed, 'rafvogel_with_joint': rafvogel_with_joint, 'orthogonality_constraint': orthogonality_constraint}}

    # get the data obj
    set_seed(seed)
    data_obj = get_dataset_obj(dataset, dataset_settings, spurious_ratio, data_info, seed, device, use_punctuation_MNLI=True)
    
    # demean, pca
    if demean:
        data_obj.demean_X(reset_mean=True, include_test=True)
    if pca:
        data_obj.transform_data_to_k_components(k_components, reset_V_k=True, include_test=True)
        V_k_train = data_obj.V_k_train

    # get the data
    X_train, y_c_train, y_m_train = data_obj.X_train, data_obj.y_c_train, data_obj.y_m_train
    X_val, y_c_val, y_m_val = data_obj.X_val, data_obj.y_c_val, data_obj.y_m_val
    X_test, y_c_test, y_m_test = data_obj.X_test, data_obj.y_c_test, data_obj.y_m_test

    # calculate the weights for the training and validation set
    if balanced_training_concept:
        concept_weights_train, concept_weights_val = data_obj.get_class_weights_train_val(y_c_train, y_c_val)
    else:
        concept_weights_train, concept_weights_val= None, None

    # If rafvogel with joint decision rule,
    if rafvogel_with_joint:
       
        if not baseline_adjust:
            print('Not the baseline difference')
            expected_diff = 0
            var_diff = 0
        # get the expected difference and variance of difference
        else:
             # define the loaders
            loaders =  data_obj.create_loaders(batch_size=batch_size, workers=0, with_concept=True, include_weights=balanced_training_concept, train_weights=concept_weights_train, val_weights=concept_weights_val)

            # Get the expected difference and variance of difference 
            expected_diff, var_diff = get_baseline_diff(data_obj, loaders, device, solver, lr,  per_step, optimizer_settings['tol'], early_stopping, optimizer_settings['patience'], epochs, bias = True)

    else:
        expected_diff, var_diff = None, None


    # Train the model to get V_c
    set_seed(seed)
    V_c_INLP, d_c_INLP = train_INLP(data_obj,
                                                                device,
                                                                batch_size=batch_size, 
                                                                solver=solver,
                                                                lr=lr,
                                                                weight_decay=weight_decay,
                                                                per_step=per_step,
                                                                tol=optimizer_settings['tol'],
                                                                early_stopping=early_stopping,
                                                                patience=optimizer_settings['patience'],
                                                                epochs=epochs,
                                                                alpha=alpha,
                                                                model_base_name='waterbird_rafvogel',
                                                                bias=True,
                                                                joint_decision_rule=rafvogel_with_joint,
                                                                include_weights=balanced_training_concept,
                                                                train_weights=concept_weights_train,
                                                                val_weights=concept_weights_val,
                                                                orthogonality_constraint=orthogonality_constraint,
                                                                expected_diff=expected_diff,
                                                                var_diff = var_diff,
                                                                )
    
    print('V_c (INLP)', V_c_INLP)

    # define orthogonal projection
    d = X_train.shape[1]
    P_c_orth = torch.eye(d) - create_P(V_c_INLP)

    # reset the data
    X_train_after_proj = torch.matmul(X_train, P_c_orth)
    X_val_after_proj = torch.matmul(X_val, P_c_orth)
    data_obj.reset_X(X_train_after_proj,X_val_after_proj, batch_size=batch_size)

    # get the main task weights
    if balanced_training_main:
        main_weights_train, main_weights_val = data_obj.get_class_weights_train_val(y_m_train, y_m_val)
    else:
        main_weights_train, main_weights_val = None, None

    # get parameters
    if use_standard_ERM_settings_after:
        lr = optimizer_info['standard_ERM_settings'][dataset]['lr']
        weight_decay = optimizer_info['standard_ERM_settings'][dataset]['weight_decay']
        batch_size = optimizer_info['standard_ERM_settings'][dataset]['batch_size']
        print('Using standard ERM settings after projecting out the concept subspace')

    # reset the data objects
    data_obj.reset_X(X_train, X_val, batch_size=batch_size, reset_X_objects=True)
    data_obj.reset_X_c_orth(P_c_orth, 
                            batch_size=batch_size, 
                            include_test=False,   
                            only_main=False,
                                                        include_weights=balanced_training_main, 
                                                        train_weights = main_weights_train, 
                                                        val_weights = main_weights_val, )

    # Train the model to get main
    set_seed(seed)


    main_model = return_linear_model(d, 
                                              data_obj.main_loader,
                                              device,
                                              solver = solver,
                                              lr=lr,
                                              per_step=per_step,
                                              tol = optimizer_settings['tol'],
                                              early_stopping = early_stopping,
                                              patience = optimizer_settings['patience'],
                                              epochs = epochs,
                                              bias=True,
                                              weight_decay=weight_decay)
    
   
    X_test_after = torch.matmul(X_test, P_c_orth)
    y_m_pred_test = main_model(X_test_after)
    main_acc_after = get_acc_pytorch_model(y_m_test, y_m_pred_test)

    # get the accuracy of the main model per group
    result_per_group, _ = get_acc_per_group(y_m_pred_test, y_m_test, y_c_test)
    result_per_group_train, _ = get_acc_per_group(main_model(X_train_after_proj), y_m_train, y_c_train)
    result_per_group_val, _ = get_acc_per_group(main_model(X_val_after_proj), y_m_val, y_c_val)

    print("Overall Accuracy of INLP: ", main_acc_after)
    print("Accuracy per group of INLP: ", result_per_group)
    print("Accuracy per group of INLP (train)", result_per_group_train)
    print("Accurafy per group of INLP (val)", result_per_group_val)
    print('Dimension of concept subspace, INLP: ',d_c_INLP)

    if dataset == 'Toy':
        print(V_c_INLP)


    if save_results:
        results_dict['overall_acc_test'] = main_acc_after  
        results_dict['result_per_group_test'] = result_per_group
        results_dict['result_per_group_train'] = result_per_group_train
        results_dict['result_per_group_val'] = result_per_group_val
        results_dict['result_per_group_test'] = result_per_group
        results_dict['weights'] = main_model.linear_layer.weight.data.detach()
        results_dict['b'] = main_model.linear_layer.bias.data.detach()
        results_dict['param_W'] = main_model.state_dict()
        results_dict['V_c'] = V_c_INLP
        results_dict['d_c'] = d_c_INLP
        results_dict['X_train'] = X_train
        results_dict['X_val'] = X_val
        results_dict['X_test'] = X_test
        results_dict['y_m_train'] = y_m_train
        results_dict['y_m_val'] = y_m_val
        results_dict['y_m_test'] = y_m_test
        results_dict['y_c_train'] = y_c_train
        results_dict['y_c_val'] = y_c_val
        results_dict['y_c_test'] = y_c_test
        
        if pca:
            results_dict['V_k_train'] = V_k_train
        if demean:
            results_dict['X_train_mean'] = data_obj.X_train_mean

         # Check whether the specified path exists or not
        folder_exists = os.path.exists('results/'+ folder)
        if not folder_exists:
            # Create a new directory because it does not exist
            os.makedirs('results/'+ folder)

       # save the results_dict in results folder
        filename = 'results/{}/{}_{}_seed_{}'.format(folder, dataset, 'INLP', seed)
        with open(filename + '.pkl', 'wb') as fp:
            pickle.dump(results_dict, fp)
            print('dictionary saved successfully to file')
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use")
    parser.add_argument("--spurious_ratio", type=float, default=0.5, help="ratio of spurious features")
    parser.add_argument("--dataset_setting", type=str, default="default", help="dataset setting to use")
    parser.add_argument("--demean", type=str,  help="whether to demean the data")
    parser.add_argument("--pca", type=str,  help="whether to use pca")
    parser.add_argument("--k_components", type=int, default=10, help="number of components to use for pca")
    parser.add_argument("--alpha", type=float, default=0.05, help="alpha to use for hypothesis testing")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size to use")
    parser.add_argument("--solver", type=str, default="SGD", help="solver to use")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate to use")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay to use")
    parser.add_argument("--early_stopping", type=str,  help="whether to use early stopping")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to use")
    parser.add_argument("--balanced_training_concept", type=str,  help="whether to use balanced training for concept")
    parser.add_argument("--balanced_training_main", type=str,  help="whether to use balanced training for main")
    parser.add_argument("--orthogonality_constraint", type=str, help = 'run Rafvogel with orthogonality constraint')
    parser.add_argument("--rafvogel_with_joint", type=str, help='Initialize rafvogel with joint decision rule')
    parser.add_argument("--per_step", type=int, default=1, help="per steps, print the loss")
    parser.add_argument("--device_type", type=str, default="cuda", help="device to use")
    parser.add_argument('--save_results')
    parser.add_argument("--seed", type=int, default=0, help="seed to use")
    parser.add_argument("--folder", type=str, help="which folder")

    args = parser.parse_args()
    dict_arguments = vars(args)

    dataset = dict_arguments["dataset"]
    spurious_ratio = dict_arguments["spurious_ratio"]
    dataset_setting = dict_arguments["dataset_setting"]
    demean = str_to_bool(dict_arguments["demean"])
    pca = str_to_bool(dict_arguments["pca"])
    k_components = dict_arguments["k_components"]
    alpha = dict_arguments["alpha"]
    batch_size = dict_arguments["batch_size"]
    solver = dict_arguments["solver"]
    lr = dict_arguments["lr"]
    weight_decay = dict_arguments["weight_decay"]
    early_stopping = str_to_bool(dict_arguments["early_stopping"])
    epochs = dict_arguments["epochs"]
    balanced_training_concept = str_to_bool(dict_arguments["balanced_training_concept"])
    balanced_training_main = str_to_bool(dict_arguments["balanced_training_main"])
    orthogonality_constraint = str_to_bool(dict_arguments["orthogonality_constraint"])
    rafvogel_with_joint = str_to_bool(dict_arguments['rafvogel_with_joint'])
    per_step = dict_arguments["per_step"]
    device_type = dict_arguments["device_type"]
    save_results = str_to_bool(dict_arguments["save_results"])
    seed = dict_arguments["seed"]
    folder = dict_arguments["folder"]
    
    main(dataset, dataset_setting, spurious_ratio, demean, pca, k_components, alpha,  batch_size, solver, lr,weight_decay,  early_stopping, epochs,balanced_training_concept, balanced_training_main, rafvogel_with_joint, orthogonality_constraint,  per_step, device_type, save_results, seed, folder)


