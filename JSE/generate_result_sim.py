
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



def main(method, sims, dataset, dataset_setting, spurious_ratio_list, demean, pca, k_components, alpha,  batch_size, solver, lr,weight_decay,  early_stopping, epochs, per_step, device_type, save_results, run_seed, group_weighted=False, null_is_concept=False, rafvogel_with_joint=False, orthogonality_constraint=False, baseline_adjust=True, delta=0.0, eval_balanced=True, acc_comparison=False, rank_RLACE=1, use_standard_settings=True, concept_first=True, remove_concept=True):

    # if dataset == 'Waterbirds', balance y_m, y_c
    if dataset == 'Waterbirds':
        balanced_training_main = True
        balanced_training_concept = True
       
    # if dataset == 'celebA', balance y_m, y_c
    elif dataset == 'celebA':
        if dataset_setting == 'sampled_data':
            balanced_training_main = False
            balanced_training_concept = False
        else:
            balanced_training_main = True
            balanced_training_concept = True
    else:
        balanced_training_main = False
        balanced_training_concept = False
    print('balanced_training_main: ', balanced_training_main)
    print('balanced_training_concept: ', balanced_training_concept)

    # create the strings for the generate_result files
    dataset_str =  '--dataset={}'.format(dataset)
    dataset_setting_str =  '--dataset_setting={}'.format(dataset_setting)
    demean_str =  '--demean={}'.format(demean)
    pca_str =  '--pca={}'.format(pca)
    k_components_str =  '--k_components={}'.format(k_components)
    alpha_str =  '--alpha={}'.format(alpha)
    batch_size_str =  '--batch_size={}'.format(batch_size)
    solver_str =  '--solver={}'.format(solver)
    lr_str =  '--lr={}'.format(lr)
    weight_decay_str =  '--weight_decay={}'.format(weight_decay)
    early_stopping_str =  '--early_stopping={}'.format(early_stopping)
    epochs_str =  '--epochs={}'.format(epochs)
    balanced_training_main_str =  '--balanced_training_main={}'.format((balanced_training_main))
    balanced_training_concept_str =  '--balanced_training_concept={}'.format(balanced_training_concept)
    per_step_str =  '--per_step={}'.format(per_step)
    device_type_str =  '--device_type={}'.format(device_type)
    save_results_str =  '--save_results={}'.format(save_results)


    # create the terminal call for ERM
    if method == 'ERM':

        if use_standard_settings and group_weighted:
            lr = optimizer_info['standard_ERM_settings_GW'][dataset]['lr']
            weight_decay = optimizer_info['standard_ERM_settings_GW'][dataset]['weight_decay']
            batch_size = optimizer_info['standard_ERM_settings_GW'][dataset]['batch_size']

        elif use_standard_settings:
            lr = optimizer_info['standard_ERM_settings'][dataset]['lr']
            weight_decay = optimizer_info['standard_ERM_settings'][dataset]['weight_decay']
            batch_size = optimizer_info['standard_ERM_settings'][dataset]['batch_size']
            print('Using standard ERM settings')

        lr_str = '--lr={}'.format(lr)
        weight_decay_str = '--weight_decay={}'.format(weight_decay)
        batch_size_str = '--batch_size={}'.format(batch_size)

        group_weighted_str =  '--group_weighted={}'.format(group_weighted)
        terminal_call = 'python3 generate_result_ERM.py ' + dataset_str + ' ' +   dataset_setting_str + ' ' + demean_str + ' ' + pca_str + ' ' + k_components_str + ' ' + alpha_str + ' ' + batch_size_str + ' ' + solver_str + ' ' + lr_str + ' ' + weight_decay_str + ' ' + early_stopping_str + ' ' + epochs_str + ' ' + balanced_training_main_str + ' ' + ' ' + group_weighted_str + ' ' + per_step_str + ' ' + device_type_str + ' ' + save_results_str 

    # create the terminal call for JSE
    elif method == 'JSE':

        if use_standard_settings:
            lr = optimizer_info['standard_JSE_settings'][dataset]['lr']
            weight_decay = optimizer_info['standard_JSE_settings'][dataset]['weight_decay']
            batch_size = optimizer_info['standard_JSE_settings'][dataset]['batch_size']
            print('Using standard JSE settings')

            lr_str = '--lr={}'.format(lr)
            weight_decay_str = '--weight_decay={}'.format(weight_decay)
            batch_size_str = '--batch_size={}'.format(batch_size)

        
        null_is_concept_str =  '--null_is_concept={}'.format(null_is_concept)
        baseline_adjust_str =  '--baseline_adjust={}'.format(baseline_adjust)
        delta_str =  '--delta={}'.format(delta)
        eval_balanced_str =  '--eval_balanced={}'.format(eval_balanced)

        concept_first_str = '--concept_first={}'.format(concept_first)
        remove_concept_str = '--remove_concept={}'.format(remove_concept)

        terminal_call = 'python generate_result_JSE.py ' + dataset_str + ' ' +  dataset_setting_str + ' ' + demean_str + ' ' + pca_str + ' ' + k_components_str + ' ' + alpha_str + ' ' + batch_size_str + ' ' + solver_str + ' ' + lr_str + ' ' + weight_decay_str + ' ' + early_stopping_str + ' ' + epochs_str + ' ' + balanced_training_main_str + ' ' + balanced_training_concept_str + ' ' + null_is_concept_str + ' ' + per_step_str + ' ' + device_type_str + ' ' + save_results_str  + ' ' + baseline_adjust_str + ' ' + delta_str + ' ' + eval_balanced_str  + ' ' + concept_first_str + ' ' + remove_concept_str

    # create the terminal call for INLP
    elif method == 'INLP':

        if use_standard_settings:
            lr = optimizer_info['standard_INLP_settings'][dataset]['lr']
            weight_decay = optimizer_info['standard_INLP_settings'][dataset]['weight_decay']
            batch_size = optimizer_info['standard_INLP_settings'][dataset]['batch_size']

            print('Using standard INLP settings')
            lr_str = '--lr={}'.format(lr)
            weight_decay_str = '--weight_decay={}'.format(weight_decay)
            batch_size_str = '--batch_size={}'.format(batch_size)
        
        rafvogel_with_joint_str =  '--rafvogel_with_joint={}'.format(rafvogel_with_joint)
        orthogonality_constraint_str =  '--orthogonality_constraint={}'.format(orthogonality_constraint)


        terminal_call = 'python generate_result_INLP.py ' + dataset_str + ' ' + dataset_setting_str  + ' ' + demean_str + ' ' + pca_str + ' ' + k_components_str + ' ' + alpha_str + ' ' + batch_size_str + ' ' + solver_str + ' ' + lr_str + ' ' + weight_decay_str + ' ' + early_stopping_str + ' ' + epochs_str + ' ' + balanced_training_main_str + ' ' + balanced_training_concept_str + ' ' + rafvogel_with_joint_str + ' ' + orthogonality_constraint_str + ' ' + per_step_str + ' ' + device_type_str + ' ' + save_results_str 


    # create the terminal call for adversarial removal
    elif method == 'ADV':
        if use_standard_settings:
            lr = optimizer_info['standard_ERM_settings'][dataset]['lr']
            weight_decay = optimizer_info['standard_ERM_settings'][dataset]['weight_decay']
            batch_size = optimizer_info['standard_ERM_settings'][dataset]['batch_size']
            
            lr_str = '--lr={}'.format(lr)
            weight_decay_str = '--weight_decay={}'.format(weight_decay)
            batch_size_str = '--batch_size={}'.format(batch_size)


        terminal_call = 'python generate_result_ADV.py ' + dataset_str + ' ' + dataset_setting_str  + ' ' + demean_str + ' ' + pca_str + ' ' + k_components_str + ' ' + alpha_str + ' ' + batch_size_str + ' ' + solver_str + ' ' + lr_str + ' ' + weight_decay_str + ' ' + early_stopping_str + ' ' + epochs_str + ' ' + balanced_training_main_str + ' '  + per_step_str + ' ' + device_type_str + ' ' + save_results_str
    
    # create the terminal call for RLACE
    elif method == 'RLACE':

        if use_standard_settings:
            lr = optimizer_info['standard_RLACE_settings'][dataset]['lr']
            weight_decay = optimizer_info['standard_RLACE_settings'][dataset]['weight_decay']
            batch_size = optimizer_info['standard_RLACE_settings'][dataset]['batch_size']
            
            lr_str = '--lr={}'.format(lr)
            weight_decay_str = '--weight_decay={}'.format(weight_decay)
            batch_size_str = '--batch_size={}'.format(batch_size)


        use_standard_INLP_settings_after = True
        use_standard_INLP_settings_after_str = '--use_standard_INLP_settings_after={}'.format(use_standard_INLP_settings_after)

        print('use standard INLP settings: ', use_standard_INLP_settings_after_str)
            

        rank_RLACE_str = '--rank={}'.format(rank_RLACE)
        terminal_call = 'python generate_result_RLACE.py ' + dataset_str + ' ' + dataset_setting_str  + ' ' + demean_str + ' ' + pca_str + ' ' + k_components_str + ' '  + batch_size_str + ' ' + solver_str + ' ' + lr_str + ' ' + weight_decay_str + ' ' + early_stopping_str + ' ' + epochs_str + ' ' + balanced_training_main_str + ' ' + balanced_training_concept_str + ' ' + per_step_str + ' ' + device_type_str + ' ' + save_results_str  + ' ' + rank_RLACE_str + ' '+use_standard_INLP_settings_after_str
    



   

    # go over each spurious ratio
    for spurious_ratio in spurious_ratio_list:

        # set the spurious_ratio_str
        spurious_ratio_str =  '--spurious_ratio={}'.format(spurious_ratio)

        # list of overall accuracy, accuracy per group
        list_overall_acc = []
        list_acc_per_group = []

        # list of V_c
        if method == 'JSE' or method == 'INLP':
            list_V_c = []
        
        if method == 'JSE':
            list_V_m = []

        # go over the sims
        for sim in range(sims):

            # set the seed_str 

            if dataset == 'multiNLI':
                seed_str = '--seed={}'.format(sim + 1)
            else:
                seed_str = '--seed={}'.format(sim + run_seed)

            # add to the terminal call
            terminal_call_sim = terminal_call + ' ' + seed_str

            # folder to save the results
            folder_name = '{}_{}_sims_{}_run_seed_{}_spurious_ratio_{}'.format(dataset, method, sims, run_seed, int(100* spurious_ratio))
            folder_name_str = '--folder={}'.format(folder_name)

            # add to the terminal call
            terminal_call_sim = terminal_call_sim + ' ' + folder_name_str

            # add the spurious ratio to the terminal call
            terminal_call_sim = terminal_call_sim + ' ' + spurious_ratio_str

           

            # run the terminal call
            print('---------------------')
            print('Now evaluating sim {}/{} at spurious ratio {} for dataset {} and method {}'.format(sim, sims, spurious_ratio, dataset, method))
            print('---------------------')

            start_time = time.time()
            os.system(terminal_call_sim)
            end_time = time.time()
            time_taken = end_time - start_time
            print('Time taken for sim: ', time_taken)
            


            # get the results from the folder
            if save_results:
                if dataset == 'multiNLI':
                    filename = '{}_{}_seed_{}'.format( dataset, method, sim + 1)
                else:
                    filename = '{}_{}_seed_{}'.format( dataset, method, sim + run_seed)
                with open('results/{}/{}.pkl'.format(folder_name, filename), 'rb') as f:
                    result_dict_for_sim = pickle.load(f)


                # get the overall accuracy
                overall_acc = result_dict_for_sim['overall_acc_test']
                list_overall_acc.append(overall_acc.item())

                # get the accuracy per group
                acc_per_group = result_dict_for_sim['result_per_group_test']
                acc_per_group['sim'] = sim
                list_acc_per_group.append(acc_per_group)

              
        

        # combine list of overall acc into dataframe with two columns; sim and overall_acc
        if save_results:
            df_overall_acc = pd.DataFrame(list_overall_acc, columns=['overall_acc'])
            df_overall_acc['sim'] = df_overall_acc.index
            print(df_overall_acc)

            # combine list of acc per group into dataframe with three columns; sim, group, and acc
            df_acc_per_group = pd.concat(list_acc_per_group)
            print(df_acc_per_group)









if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="JSE", help="method to use")
    parser.add_argument("--sims", type=int, default=10, help="number of simulations to run")
    parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use")
    parser.add_argument("--spurious_ratio", help="ratio of spurious features")
    parser.add_argument("--dataset_setting", type=str, default="default", help="dataset setting to use")
    parser.add_argument("--demean", type=str,  help="whether to demean the data")
    parser.add_argument("--pca", type=str, default=False, help="whether to use pca")
    parser.add_argument("--k_components", type=int, default=10, help="number of components to use for pca")
    parser.add_argument("--alpha", type=float, default=0.05, help="alpha to use for hypothesis testing")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size to use")
    parser.add_argument("--solver", type=str, default="SGD", help="solver to use")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate to use")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay to use")
    parser.add_argument("--early_stopping", type=str, help="whether to use early stopping")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to use")
    parser.add_argument("--per_step", type=int, default=1, help="per steps, print the loss")
    parser.add_argument("--device_type", type=str, default="cuda", help="device to use")
    parser.add_argument("--save_results",type=str )
    parser.add_argument("--run_seed", type=int, default=0, help="seed to use")

    
    parser.add_argument("--null_is_concept", type=str, default='False', help="whether to use group weighted loss")
    parser.add_argument("--baseline_adjust", type=str, default='False', help="whether to use group weighted loss")
    parser.add_argument('--group_weighted', type=str, default='False', help='whether to use group weighted loss')    
    parser.add_argument('--rafvogel_with_joint', type=str, default='False', help='whether to use group weighted loss')
    parser.add_argument('--orthogonality_constraint', type=str, default='False', help='whether to use group weighted loss')
    parser.add_argument('--delta', type=float, default=0.0, help='delta for hypothesis testing')
    parser.add_argument("--eval_balanced", type=str, default="True", help="whether to use balanced evaluation")
    parser.add_argument("--concept_first", type=str, default="True", help="whether to use concept first")
    parser.add_argument("--remove_concept", type=str, default="True", help="whether to remove concept")

    parser.add_argument("--rank_RLACE", type=int, default=0, help="rank for RLACE")
    parser.add_argument("--use_standard_settings", type=str, default="True", help="whether to use standard settings")


    args = parser.parse_args()
    dict_arguments = vars(args)

    method = dict_arguments["method"]
    sims = dict_arguments["sims"]
    dataset = dict_arguments["dataset"]
    spurious_ratio = Convert(dict_arguments["spurious_ratio"], float)
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
    per_step = dict_arguments["per_step"]
    device_type = dict_arguments["device_type"]
    save_results = dict_arguments["save_results"]
    run_seed = dict_arguments["run_seed"]

    null_is_concept = str_to_bool(dict_arguments["null_is_concept"])
    baseline_adjust = str_to_bool(dict_arguments["baseline_adjust"])
    group_weighted = str_to_bool(dict_arguments["group_weighted"])
    rafvogel_with_joint = str_to_bool(dict_arguments["rafvogel_with_joint"])
    orthogonality_constraint = str_to_bool(dict_arguments["orthogonality_constraint"])
    delta = dict_arguments["delta"]
    eval_balanced = str_to_bool(dict_arguments["eval_balanced"])
    concept_first = str_to_bool(dict_arguments["concept_first"])
    remove_concept = str_to_bool(dict_arguments["remove_concept"])

    use_standard_settings = str_to_bool(dict_arguments["use_standard_settings"])

    rank_RLACE = dict_arguments["rank_RLACE"]
    

    

   

    main(method, sims, dataset, dataset_setting, spurious_ratio, demean, pca, k_components, alpha,  batch_size, solver, lr,weight_decay,  early_stopping, epochs, per_step, device_type, save_results, run_seed, null_is_concept=null_is_concept, group_weighted=group_weighted, rafvogel_with_joint=rafvogel_with_joint, orthogonality_constraint=orthogonality_constraint, baseline_adjust=baseline_adjust, delta=delta, eval_balanced=eval_balanced,  rank_RLACE=rank_RLACE, use_standard_settings=use_standard_settings, concept_first=concept_first, remove_concept=remove_concept)
