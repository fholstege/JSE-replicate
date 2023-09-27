


from JSE.RLACE import *
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


def main(dataset, dataset_setting,spurious_ratio, demean, pca, k_components,rank,  solver, lr, batch_size, per_step, early_stopping,  epochs, weight_decay, balanced_training_main, balanced_training_concept, device_type, save_results, seed, folder, num_iters = 50000, optimizer_class = torch.optim.SGD,   epsilon = 0.01, use_standard_ERM_settings_after = True, use_standard_INLP_settings_after=True):

  
    # make sure RLACE is used for same predictor
    if use_standard_INLP_settings_after:
        optimizer_params_predictor = {"lr":  optimizer_info['standard_INLP_settings'][dataset]['lr'],
                                      "weight_decay":  optimizer_info['standard_INLP_settings'][dataset]['weight_decay']}
    else:
        optimizer_params_predictor = {"lr": lr,"weight_decay": weight_decay}
    optimizer_params_P = {"lr": lr, "weight_decay": weight_decay}
    batch_size_P = batch_size

    print('optimizer_params_predictor', optimizer_params_predictor)
    print('optimizer_params_P', optimizer_params_P)

    # save all the results in a dict for this run
    results_dict = { 'method': 'RLACE', 'parameters' :{'dataset': dataset,'spurious_ratio': spurious_ratio,  'demean': demean, 'pca': pca, 'k_components': k_components,  'batch_size': batch_size, 'solver': solver, 'lr': lr, 'weight_decay': weight_decay, 'early_stopping': early_stopping, 'epochs': epochs, 'balanced_training_main': balanced_training_main, 'per_step': per_step, 'device_type': device_type,  'seed': seed}}


    # set the device
    device = torch.device(device_type)

    # set the settings for dataset
    dataset_settings = data_info[dataset][dataset_setting]
    optimizer_settings = optimizer_info['All']

    # get the data obj, set seed
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

    set_seed(seed)
    print('at this point in the script, balanced_training_concept is ', balanced_training_concept)
    output = solve_adv_game(X_train, y_c_train, X_val, y_c_val, rank=rank, device="cpu", out_iters=num_iters, optimizer_class=optimizer_class, optimizer_params_P =optimizer_params_P, optimizer_params_predictor=optimizer_params_predictor, epsilon=epsilon,batch_size=batch_size_P, balance_y=balanced_training_concept)

    P_svd = torch.Tensor(output["P"])

    # project out the subspace
    X_train_after_RLACE = torch.matmul(X_train, P_svd)
    X_val_after_RLACE = torch.matmul(X_val, P_svd)
    X_test_after_RLACE = torch.matmul(X_test, P_svd)

   
    # reset_X with the training/validation data
    if balanced_training_main:
        main_weights_train, main_weights_val = data_obj.get_class_weights_train_val(y_m_train, y_m_val)
    else:
        main_weights_train, main_weights_val = None, None

    # reset_X with the training/validation data
    if use_standard_ERM_settings_after:
        lr = optimizer_info['standard_ERM_settings'][dataset]['lr']
        weight_decay = optimizer_info['standard_ERM_settings'][dataset]['weight_decay']
        batch_size = optimizer_info['standard_ERM_settings'][dataset]['batch_size']
        print('Using standard ERM settings after projecting out the concept subspace')



    # reset_X with the training/validation data
    data_obj.reset_X(X_train_after_RLACE, X_val_after_RLACE, batch_size=batch_size, include_weights=balanced_training_main,train_weights = main_weights_train, val_weights = main_weights_val, only_main=True)


    
     # Train the model to get main
    set_seed(seed)
    main_model = return_linear_model(X_train_after_RLACE.shape[1], 
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
    
    # get the accuracy of the main model
    y_m_pred_test = main_model(X_test_after_RLACE)
    y_m_pred_train = main_model(X_train_after_RLACE)
    y_m_pred_val = main_model(X_val_after_RLACE)
    main_acc_after = get_acc_pytorch_model(y_m_test, y_m_pred_test)

    # get the accuracy of the main model per group
    result_per_group, _ = get_acc_per_group(y_m_pred_test, y_m_test, y_c_test)
    result_per_group_train, _ = get_acc_per_group(y_m_pred_train, y_m_train, y_c_train)
    result_per_group_val, _ = get_acc_per_group(y_m_pred_val, y_m_val, y_c_val)

    print("Overall Accuracy of RLACE: ", main_acc_after)
    print("Accuracy per group of RLACE: ", result_per_group)



    if save_results:
        results_dict['overall_acc_test'] = main_acc_after  
        results_dict['result_per_group_test'] = result_per_group
        results_dict['result_per_group_train'] = result_per_group_train
        results_dict['result_per_group_val'] = result_per_group_val
        results_dict['weights'] = main_model.linear_layer.weight.data.detach()
        results_dict['b'] = main_model.linear_layer.bias.data.detach()
        results_dict['param_W'] = main_model.state_dict()
        results_dict['P_c_orth'] = P_svd
        results_dict['d_c'] =rank

        if pca:
            results_dict['V_k_train'] = V_k_train

         # Check whether the specified path exists or not
        folder_exists = os.path.exists('results/'+ folder)
        if not folder_exists:
            # Create a new directory because it does not exist
            os.makedirs('results/'+ folder)

        # save the results_dict in results folder
        filename = 'results/{}/{}_{}_seed_{}_eps_{}'.format(folder, dataset, 'RLACE', seed, epsilon)
        with open(filename + '.pkl', 'wb') as fp:
            pickle.dump(results_dict, fp)
            print('dictionary saved successfully to file')
            




                        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Toy', help='dataset to use')
    parser.add_argument('--dataset_setting', type=str, default='Toy', help='dataset setting to use')
    parser.add_argument('--spurious_ratio', type=float, default=0.5, help='spurious ratio to use')
    parser.add_argument('--demean', type=bool, default=False, help='demean the data')
    parser.add_argument('--pca', type=str, help='pca the data')
    parser.add_argument('--k_components', type=int, default=2, help='number of components to use for pca')
    parser.add_argument('--rank', type=int, default=1, help='rank to use')
    parser.add_argument('--solver', type=str, default='lbfgs', help='solver to use')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate to use')
    parser.add_argument('--batch_size', type=int, help='batch size to use')
    parser.add_argument('--per_step', type=int, default=10, help='per step to use')
    parser.add_argument('--early_stopping',  help='early stopping to use')
    parser.add_argument('--epochs', type=int, default=100, help='epochs to use')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay to use')
    parser.add_argument('--balanced_training_concept', type=str, help='balanced training concept to use')
    parser.add_argument('--balanced_training_main', type=str,  help='balanced training main to use')
    parser.add_argument('--seed', type=int, default=0, help='seed to use')
    parser.add_argument('--device_type', type=str, default='cpu', help='device type to use')
    parser.add_argument('--folder', type=str, default='results', help='folder to use')
    parser.add_argument('--save_results', type=str, help='save results to use')
    parser.add_argument('--use_standard_INLP_settings_after', type=str, help='use standard INLP settings after to use')


    args = parser.parse_args()
    dict_arguments = vars(args)
    print(dict_arguments.keys())
    print(dict_arguments)

    dataset  = dict_arguments['dataset']
    dataset_setting = dict_arguments['dataset_setting']
    spurious_ratio = dict_arguments['spurious_ratio']
    demean = dict_arguments['demean']
    pca = str_to_bool(dict_arguments['pca'])
    k_components = dict_arguments['k_components']
    rank = dict_arguments['rank']
    solver = dict_arguments['solver']
    lr = dict_arguments['lr']
    batch_size = dict_arguments['batch_size']
    per_step = dict_arguments['per_step']
    early_stopping = str_to_bool(dict_arguments['early_stopping'])
    epochs = dict_arguments['epochs']
    weight_decay = dict_arguments['weight_decay']
    balanced_training_concept = str_to_bool(dict_arguments['balanced_training_concept'])
    balanced_training_main = str_to_bool(dict_arguments['balanced_training_main'])
    seed = dict_arguments['seed']
    device_type = dict_arguments['device_type']
    folder = dict_arguments['folder']
    save_results = str_to_bool(dict_arguments['save_results'])
    use_standard_INLP_settings_after = str_to_bool(dict_arguments['use_standard_INLP_settings_after'])

    main(dataset, dataset_setting,spurious_ratio, demean, pca, k_components,rank,  solver, lr, batch_size, per_step, early_stopping, epochs, weight_decay,balanced_training_main, balanced_training_concept, device_type, save_results, seed, folder, use_standard_INLP_settings_after=use_standard_INLP_settings_after)