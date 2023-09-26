
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



def main(methods, sims, dataset, dataset_setting, spurious_ratio_str, demean, pca, k_components, alpha, run_seed, null_is_concept=False, group_weighted=False, rafvogel_with_joint=False, orthogonality_constraint=True, baseline_adjust=False, delta=0, eval_balanced=True, rank_RLACE=1, remove_concept=True, concept_first=True):


    
    # create the strings for the generate_result files
    dataset_str =  '--dataset={}'.format(dataset)
    dataset_setting_str =  '--dataset_setting={}'.format(dataset_setting)
    demean_str =  '--demean={}'.format(demean)
    pca_str =  '--pca={}'.format(pca)
    k_components_str =  '--k_components={}'.format(k_components)
    alpha_str =  '--alpha={}'.format(alpha)
    sims_str =  '--sims={}'.format(sims)
    run_seed_str =  '--run_seed={}'.format(run_seed)


    use_standard_settings_str = '--use_standard_settings={}'.format('True')
    solver_str = '--solver={}'.format('SGD')
    early_stopping_str = '--early_stopping={}'.format('True')
    epochs_str = '--epochs={}'.format(50)
    per_step_str = '--per_step={}'.format(5)
    device_type_str = '--device_type={}'.format('cpu')
    save_results_str = '--save_results={}'.format('True')
    spurious_ratio_str = '--spurious_ratio={}'.format(spurious_ratio_str)


    

    for method in methods:
        print('method: ', method)

        method_str = '--method={}'.format(method)


        # create the terminal call for ERM
        if method == 'ERM':

            lr = optimizer_info['standard_ERM_settings'][dataset]['lr']
            weight_decay = optimizer_info['standard_ERM_settings'][dataset]['weight_decay']
            batch_size = optimizer_info['standard_ERM_settings'][dataset]['batch_size']

            lr_str = '--lr={}'.format(lr)
            weight_decay_str = '--weight_decay={}'.format(weight_decay)
            batch_size_str = '--batch_size={}'.format(batch_size)
            group_weighted_str = '--group_weighted={}'.format(group_weighted)



            group_weighted_str =  '--group_weighted={}'.format(group_weighted)

            terminal_call = 'python3 generate_result_sim.py ' + method_str + ' ' + dataset_str + ' ' +   dataset_setting_str + ' ' + demean_str + ' ' + pca_str + ' ' + k_components_str + ' ' + alpha_str + ' ' + batch_size_str + ' ' + solver_str + ' ' + lr_str + ' ' + weight_decay_str + ' ' + early_stopping_str + ' ' + epochs_str + ' '  + group_weighted_str + ' ' + per_step_str + ' ' + device_type_str + ' ' + save_results_str  + ' ' + use_standard_settings_str + ' ' + sims_str + ' ' + spurious_ratio_str + ' ' + run_seed_str

        # create the terminal call for JSE
        elif method == 'JSE':

            lr = optimizer_info['standard_JSE_settings'][dataset]['lr']
            weight_decay = optimizer_info['standard_JSE_settings'][dataset]['weight_decay']
            batch_size = optimizer_info['standard_JSE_settings'][dataset]['batch_size']

            lr_str = '--lr={}'.format(lr)
            weight_decay_str = '--weight_decay={}'.format(weight_decay)
            batch_size_str = '--batch_size={}'.format(batch_size)

            
            null_is_concept_str =  '--null_is_concept={}'.format(null_is_concept)
            baseline_adjust_str =  '--baseline_adjust={}'.format(baseline_adjust)
            delta_str =  '--delta={}'.format(delta)
            eval_balanced_str =  '--eval_balanced={}'.format(eval_balanced)
            concept_first_str = '--concept_first={}'.format(concept_first)
            remove_concept_str = '--remove_concept={}'.format(remove_concept)

            terminal_call = 'python generate_result_sim.py '+ method_str + ' ' + dataset_str + ' ' +  dataset_setting_str + ' ' + demean_str + ' ' + pca_str + ' ' + k_components_str + ' ' + alpha_str + ' ' + batch_size_str + ' ' + solver_str + ' ' + lr_str + ' ' + weight_decay_str + ' ' + early_stopping_str + ' ' + epochs_str +  ' ' + null_is_concept_str + ' ' + per_step_str + ' ' + device_type_str + ' ' + save_results_str  + ' ' + baseline_adjust_str + ' ' + delta_str + ' ' + eval_balanced_str + ' ' + use_standard_settings_str + ' ' + sims_str + ' ' + spurious_ratio_str + ' ' + run_seed_str + ' ' + concept_first_str + ' ' + remove_concept_str

        # create the terminal call for INLP
        elif method == 'INLP':

            lr = optimizer_info['standard_INLP_settings'][dataset]['lr']
            weight_decay = optimizer_info['standard_INLP_settings'][dataset]['weight_decay']
            batch_size = optimizer_info['standard_INLP_settings'][dataset]['batch_size']

            lr_str = '--lr={}'.format(lr)
            weight_decay_str = '--weight_decay={}'.format(weight_decay)
            batch_size_str = '--batch_size={}'.format(batch_size)
            
            rafvogel_with_joint_str =  '--rafvogel_with_joint={}'.format(rafvogel_with_joint)
            orthogonality_constraint_str =  '--orthogonality_constraint={}'.format(orthogonality_constraint)


            terminal_call = 'python generate_result_sim.py ' + method_str+ ' ' +  dataset_str + ' ' + dataset_setting_str  + ' ' + demean_str + ' ' + pca_str + ' ' + k_components_str + ' ' + alpha_str + ' ' + batch_size_str + ' ' + solver_str + ' ' + lr_str + ' ' + weight_decay_str + ' ' + early_stopping_str + ' ' + epochs_str + ' ' + rafvogel_with_joint_str + ' ' + orthogonality_constraint_str + ' ' + per_step_str + ' ' + device_type_str + ' ' + save_results_str + ' ' + use_standard_settings_str + ' ' + sims_str + ' ' + spurious_ratio_str + ' ' + run_seed_str


        # create the terminal call for adversarial removal
        elif method == 'ADV':
            lr = optimizer_info['standard_ERM_settings'][dataset]['lr']
            weight_decay = optimizer_info['standard_ERM_settings'][dataset]['weight_decay']
            batch_size = optimizer_info['standard_ERM_settings'][dataset]['batch_size']
                
            lr_str = '--lr={}'.format(lr)
            weight_decay_str = '--weight_decay={}'.format(weight_decay)
            batch_size_str = '--batch_size={}'.format(batch_size)


            terminal_call = 'python generate_result_sim.py ' + method_str+ ' ' + dataset_str + ' ' + dataset_setting_str  + ' ' + demean_str + ' ' + pca_str + ' ' + k_components_str + ' ' + alpha_str + ' ' + batch_size_str + ' ' + solver_str + ' ' + lr_str + ' ' + weight_decay_str + ' ' + early_stopping_str + ' ' + epochs_str + ' '  + per_step_str + ' ' + device_type_str + ' ' + save_results_str + ' ' + use_standard_settings_str + ' ' + sims_str + ' ' + spurious_ratio_str + ' ' + run_seed_str
        
        # create the terminal call for RLACE
        elif method == 'RLACE':


            lr = optimizer_info['standard_RLACE_settings'][dataset]['lr']
            weight_decay = optimizer_info['standard_RLACE_settings'][dataset]['weight_decay']
            batch_size = optimizer_info['standard_RLACE_settings'][dataset]['batch_size']
                
            lr_str = '--lr={}'.format(lr)
            weight_decay_str = '--weight_decay={}'.format(weight_decay)
            batch_size_str = '--batch_size={}'.format(batch_size)

            rank_RLACE_str = '--rank_RLACE={}'.format(rank_RLACE)
                

            terminal_call = 'python generate_result_sim.py ' + method_str+ ' '+ dataset_str + ' ' + dataset_setting_str  + ' ' + demean_str + ' ' + pca_str + ' ' + k_components_str + ' '  + batch_size_str + ' ' + solver_str + ' ' + lr_str + ' ' + weight_decay_str + ' ' + early_stopping_str + ' ' + epochs_str +  ' ' + per_step_str + ' ' + device_type_str + ' ' + save_results_str  + ' ' + rank_RLACE_str + ' ' + use_standard_settings_str + ' ' + sims_str + ' ' + spurious_ratio_str + ' ' + run_seed_str + ' ' + rank_RLACE_str
        
    
        # create the terminal call
        os.system(terminal_call)



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
    parser.add_argument('--rank_RLACE', type=int, default=1, help='rank for RLACE')

    parser.add_argument('--concept_first', type=str, default='True', help='whether to use group weighted loss')
    parser.add_argument('--remove_concept', type=str, default='True', help='whether to use group weighted loss')




    args = parser.parse_args()
    dict_arguments = vars(args)

    method = Convert(dict_arguments["method"], str)
    sims = dict_arguments["sims"]
    dataset = dict_arguments["dataset"]
    spurious_ratio = dict_arguments["spurious_ratio"]
    dataset_setting = dict_arguments["dataset_setting"]
    demean = str_to_bool(dict_arguments["demean"])
    pca = str_to_bool(dict_arguments["pca"])
    k_components = dict_arguments["k_components"]
    alpha = dict_arguments["alpha"]
    run_seed = dict_arguments["run_seed"]

    null_is_concept = str_to_bool(dict_arguments["null_is_concept"])
    baseline_adjust = str_to_bool(dict_arguments["baseline_adjust"])
    group_weighted = str_to_bool(dict_arguments["group_weighted"])
    rafvogel_with_joint = str_to_bool(dict_arguments["rafvogel_with_joint"])
    orthogonality_constraint = str_to_bool(dict_arguments["orthogonality_constraint"])
    delta = dict_arguments["delta"]
    eval_balanced = str_to_bool(dict_arguments["eval_balanced"])
    rank_RLACE = int(dict_arguments["rank_RLACE"])
    concept_first = str_to_bool(dict_arguments["concept_first"])
    remove_concept = str_to_bool(dict_arguments["remove_concept"])

    

    

   

    main(method, sims, dataset, dataset_setting, spurious_ratio, demean, pca, k_components, alpha, run_seed, null_is_concept=null_is_concept, group_weighted=group_weighted, rafvogel_with_joint=rafvogel_with_joint, orthogonality_constraint=orthogonality_constraint, baseline_adjust=baseline_adjust, delta=delta, eval_balanced=eval_balanced, rank_RLACE=rank_RLACE, concept_first=concept_first, remove_concept=remove_concept)
