


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


def main(method, param_name, sims, dataset, spurious_ratio, dataset_setting, demean, pca, k_components,   batch_size, solver, lr, weight_decay, early_stopping, epochs,  per_step, device_type, start_seed, balanced_training_main=False, balanced_training_concept=False,eval_balanced=True, metric='BCE', group_weighted=False):

    # create all the combinations between the different values in weight_decay, lr, batch_size
    combinations = []

    if dataset == 'Toy':
        balanced_training_concept = False
        balanced_training_main = False
    elif dataset == 'celebA':
        balanced_training_concept = False
        balanced_training_main = False

    # check if weight decay is a list, if not make it
    if not isinstance(weight_decay, list):
        weight_decay = [weight_decay]
    if not isinstance(lr, list):
        lr = [lr]
    if not isinstance(batch_size, list):
        batch_size = [batch_size]

    for weight_decay_i in weight_decay:
        for lr_i in lr:
            for batch_size_i in batch_size:
                combination_i = {'weight_decay': weight_decay_i, 'lr': lr_i, 'batch_size': batch_size_i}
                combinations.append(combination_i)


    # create a dict with the combinations
    n_combinations = len(combinations)
    combinations_dict = {i: {'param': combinations[i]} for i in range(0, n_combinations)}
    
    # loop over each param type
    i = 0
    for combination in combinations:
        
        # get the parameters
        weight_decay_i = combination['weight_decay']
        lr_i = combination['lr']
        batch_size_i = combination['batch_size']
        print('Checking for combination: weight_decay: {}, lr: {}, batch_size: {}'.format(weight_decay_i, lr_i, batch_size_i))
        combinations_dict[i]['results'] = [None]*sims


        # loop over the simulations
        for run_i in range(sims):
                
                # set the seed per run
                seed = start_seed + run_i
                
                # set the device
                device = torch.device(device_type)

                # set the settings for dataset
                dataset_settings = data_info[dataset][dataset_setting]
                optimizer_settings = optimizer_info['All']

            
                # get the data obj
                set_seed(seed)
                data_obj = get_dataset_obj(dataset, dataset_settings, spurious_ratio, data_info, seed, device, use_punctuation_MNLI=True)
                
                
                # demean and pca
                if pca:
                    data_obj.transform_data_to_k_components(k_components, reset_V_k=True, include_test=True)
                    V_k_train = data_obj.V_k_train

                if demean:
                    data_obj.demean_X(reset_mean=True, include_test=True)


                X_train, y_c_train, y_m_train = data_obj.X_train, data_obj.y_c_train, data_obj.y_m_train
                X_val, y_c_val, y_m_val = data_obj.X_val, data_obj.y_c_val, data_obj.y_m_val

                
                if dataset == 'Waterbirds' and dataset_setting == 'default':
                    balanced_training_concept = True
                    balanced_training_main = True
               

                # calculate the weights for the training and validation set
                if balanced_training_concept:
                    concept_weights_train, concept_weights_val = data_obj.get_class_weights_train_val(y_c_train, y_c_val)
                else:
                    concept_weights_train, concept_weights_val= None, None

                # calculate the weights for the training and validation set
                if balanced_training_main:
                    main_weights_train, main_weights_val= data_obj.get_class_weights_train_val(y_m_train, y_m_val)
                elif group_weighted:
                    main_weights_train, main_weights_val = data_obj.get_group_weights()
                else:
                    main_weights_train, main_weights_val= None, None


                if method == 'JSE':

                    # define the loaders
                    loaders =  data_obj.create_loaders(batch_size=batch_size_i, workers=0, with_concept=True, include_weights=balanced_training_concept, train_weights=concept_weights_train, val_weights=concept_weights_val)


                     # get the model
                    d = data_obj.X_train.shape[1]
                    joint_model = return_joint_main_concept_model(d,  
                                                  loaders, 
                                                  device,
                                                               solver=solver,
                                                               lr=lr_i,
                                                               weight_decay=weight_decay_i,
                                                               per_step=per_step,
                                                               tol=optimizer_settings['tol'],
                                                               early_stopping=early_stopping,
                                                               patience=optimizer_settings['patience'],
                                                               epochs=epochs,
                                                               bias=True,
                                                               model_name = 'select_param_JSE_'+dataset,
                                                               save_best_model=True
                                                               )
                    
                    # calculate the binary cross-entropy of the concept model
                    X_val = data_obj.X_val
                    output = joint_model(X_val)
                    y_c_val_pred = output['y_c_1']
                    y_m_val_pred = output['y_m_1']

                    # calculate the binary cross-entropy of the models
                    if eval_balanced:
                            BCE_concept = torch_calc_BCE_weighted(
                                        y_m_val, y_c_val, y_c_val_pred, device, reduce='mean', type_dependent='concept')
                            BCE_main = torch_calc_BCE_weighted(
                                        y_m_val, y_c_val, y_m_val_pred, device, reduce='mean', type_dependent='main')
                    else:
                            BCE_concept = torch_calc_BCE(y_c_val,y_c_val_pred, device, reduce='mean')
                            BCE_main = torch_calc_BCE(y_m_val,y_m_val_pred, device, reduce='mean')

                    print('BCE concept: ', BCE_concept.detach().item())
                    print('BCE main: ', BCE_main.detach().item())
                    combined_loss = (BCE_concept.detach().item() + BCE_main.detach().item())/2 
                   
                    print('Simulation {} of {} for comibination: weight_decay: {}, lr: {}, batch_size: {}'.format(run_i, sims, weight_decay_i, lr_i, batch_size_i))
                   
                   
                    combinations_dict[i]['results'][run_i] = combined_loss

                elif method == 'INLP':
                    
                    # define the loaders
                    d = data_obj.X_train.shape[1]
                    concept_loader =  data_obj.create_loaders(batch_size=batch_size_i, workers=0, with_concept=False,which_dependent='concept', include_weights=balanced_training_concept, train_weights=concept_weights_train, val_weights=concept_weights_val)


                    # get the model
                    concept_model = return_linear_model(d, 
                                                concept_loader,
                                                device,
                                                solver = solver,
                                                lr=lr_i,
                                                per_step=per_step,
                                                tol = optimizer_settings['tol'],
                                                early_stopping = early_stopping,
                                                patience = optimizer_settings['patience'],
                                                epochs = epochs,
                                                bias=True,
                                                weight_decay=weight_decay_i, 
                                                model_name=dataset+'_select_param_concept_model')
                    
                    # calculate the binary cross-entropy of the concept model
                    X_val = data_obj.X_val
                    y_c_val = data_obj.y_c_val
                    y_c_val_pred = concept_model(X_val)


                    if metric == 'WG':

                        accuracy_per_group, _ =  get_acc_per_group(y_c_val_pred, y_c_val, y_m_val)
                        worst_group_accuracy = np.min(accuracy_per_group)
                        combinations_dict[i]['results'][run_i] = -worst_group_accuracy
                       
                    else:

                        if eval_balanced:
                            BCE_concept = torch_calc_BCE_weighted(y_m_val, y_c_val,y_c_val_pred, device, type_dependent='concept', reduce='mean')
                        else:
                            BCE_concept = torch_calc_BCE(y_c_val,y_c_val_pred, device)
                        combinations_dict[i]['results'][run_i] = BCE_concept.detach().item()

                
                elif method == 'ERM':
                    
                    # define the loaders
                    d = data_obj.X_train.shape[1]
                    main_loader =  data_obj.create_loaders(batch_size=batch_size_i, workers=0, with_concept=False,which_dependent='main', include_weights=balanced_training_main, train_weights=main_weights_train, val_weights=main_weights_val)


                    # get the model
                    main_model = return_linear_model(d, 
                                                main_loader,
                                                device,
                                                solver = solver,
                                                lr=lr_i,
                                                per_step=per_step,
                                                tol = optimizer_settings['tol'],
                                                early_stopping = early_stopping,
                                                patience = optimizer_settings['patience'],
                                                epochs = epochs,
                                                bias=True,
                                                weight_decay=weight_decay_i, 
                                                model_name=dataset+'_select_param_main_model')
                    
                    # calculate the binary cross-entropy of the concept model
                    X_val = data_obj.X_val
                    y_m_val = data_obj.y_m_val
                    y_m_val_pred = main_model(X_val)

                    if metric == 'WG':
                        accuracy_per_group, _ =  get_acc_per_group(y_m_val_pred, y_m_val, y_c_val)
                        worst_group_accuracy = np.min(accuracy_per_group)
                        combinations_dict[i]['results'][run_i] = -worst_group_accuracy

                    else:

                        if eval_balanced:
                            BCE_main = torch_calc_BCE_weighted(y_m_val, y_c_val,y_m_val_pred, device, type_dependent='main', reduce='mean')
                        else:
                            BCE_main = torch_calc_BCE(y_m_val,y_m_val_pred, device)
                        combinations_dict[i]['results'][run_i] = BCE_main.detach().item()


        i += 1
      

    

        

    # print the results at end
    print('-----------------------------------')

    overall_results = {}
    best_combination = None
    best_score = math.inf
    one_se_rule_combination = None
    best_score_plus_se = math.inf
    score_of_one_se_rule = math.inf
    weight_decay_best = None
    learning_rate_best = None
    batch_size_best = None



    for i in range(n_combinations):
        result_combination = combinations_dict[i]
        weight_decay_i = result_combination['param']['weight_decay']

        print('Average of BCE concept/main: ', np.mean(result_combination['results']))
        print('Standard deviation of BCE concept/main: ', np.std(result_combination['results']))
        print('For combination: {}'.format(result_combination['param']))

        avg_score = np.mean(result_combination['results'])
        std_score = np.std(result_combination['results'])
        se_score = std_score/np.sqrt(sims)

        if avg_score < best_score:
            best_combination = result_combination['param']
            best_score = avg_score
            best_score_plus_se = avg_score + se_score
            weight_decay_best = weight_decay_i
            learning_rate_best = lr_i
            print('current best score combination: {}'.format(best_combination))

        
        if (batch_size_i == batch_size_best) and (lr_i == learning_rate_best) and (weight_decay_i > weight_decay_best):

            if avg_score < best_score_plus_se and avg_score > best_score:
                print('-----------------------------------')
                print('combination based on one standard error rule')
                print(result_combination['param'])
                print('-----------------------------------')
                one_se_rule_combination = result_combination['param']
                score_of_one_se_rule = avg_score 

        

        entry = {'parameters': result_combination['param'], 'avg': avg_score, 'std': std_score, 'se': se_score}
        overall_results[i] = entry

    print('-----------------------------------')
    print('Overall results: {}'.format(overall_results))
    

    if one_se_rule_combination is None:
        print('No one standard error rule combination found - no other combination within one SE and higher weight_decay ')
    print('-----------------------------------')
    print('Best combination: {}'.format(best_combination))
    print('Best score: {}'.format(best_score))
    print('-----------------------------------')
    print('One standard error rule combination: {}'.format(one_se_rule_combination))
    print('Best score plus standard error: {}'.format(score_of_one_se_rule))



          
        






if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default='JSE', help='Method to use')
    parser.add_argument("--param_name", default="lr", help="dataset to use")
    parser.add_argument("--spurious_ratio", type=float, default=0.5, help="ratio of spurious features")
    parser.add_argument("--sims", type=int, default=10, help="number of simulations to run")
    parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use")
    parser.add_argument("--dataset_setting", type=str, default="default", help="dataset setting to use")
    parser.add_argument("--demean", type=str,  help="whether to demean the data")
    parser.add_argument("--pca", type=str, default=False, help="whether to use pca")
    parser.add_argument("--k_components", type=int, default=10, help="number of components to use for pca")
    parser.add_argument("--batch_size", help="batch size to use")
    parser.add_argument("--solver", type=str, default="SGD", help="solver to use")
    parser.add_argument("--lr",  help="learning rate to use")
    parser.add_argument("--weight_decay",  help="weight decay to use")
    parser.add_argument("--early_stopping", type=str, help="whether to use early stopping")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to use")
    parser.add_argument("--per_step", type=int, default=1, help="per steps, print the loss")
    parser.add_argument("--seed", type=int, default=0, help="seed to use")
    parser.add_argument("--device_type", type=str, default="cuda", help="device to use")
    parser.add_argument("--group_weighted", type=str, default='False', help="whether to use group-weighted for main task")

    
    args = parser.parse_args()
    dict_arguments = vars(args)

    method = dict_arguments['method']
    param_name = Convert(dict_arguments["param_name"], str)


    sims = dict_arguments["sims"]
    dataset = dict_arguments["dataset"]
    spurious_ratio = dict_arguments["spurious_ratio"]
    dataset_setting = dict_arguments["dataset_setting"]
    demean = str_to_bool(dict_arguments["demean"])
    pca = str_to_bool(dict_arguments["pca"])
    k_components = dict_arguments["k_components"]

    
    solver = dict_arguments["solver"]

    if 'weight_decay' in param_name:
        weight_decay = Convert(dict_arguments["weight_decay"], float)
    else:
        weight_decay = float(dict_arguments["weight_decay"])

    if 'batch_size' in param_name:
        batch_size = Convert(dict_arguments["batch_size"], int)
    else:
        batch_size = int(dict_arguments["batch_size"])

    if 'lr' in param_name:
        lr = Convert(dict_arguments["lr"], float)
    else:
        lr = float(dict_arguments["lr"])
    
    early_stopping = str_to_bool(dict_arguments["early_stopping"])
    epochs = dict_arguments["epochs"]
    per_step = dict_arguments["per_step"]
    device_type = dict_arguments["device_type"]
    seed = dict_arguments["seed"]
    group_weighted = str_to_bool(dict_arguments["group_weighted"])

    main(method, param_name, sims, dataset, spurious_ratio, dataset_setting, demean, pca, k_components,   batch_size, solver, lr, weight_decay, early_stopping, epochs,  per_step, device_type, seed, group_weighted=group_weighted)
