


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
import seaborn as sns
import re 
import matplotlib as mpl
import tikzplotlib as tpl
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.ticker import StrMethodFormatter, PercentFormatter
import tikzplotlib
from tikzplotlib import save as tikz_save


def main(datasets, methods, list_spurious_ratio, run_seed, sims, plot_type, mode, save_result, plot_index, vector_index, project_out_concept_vector, group_weighted_ERM=False):

    if len(datasets) == 1:
        dataset = datasets[0]
    


    # gather all the results; for given dataset, per method, for all available spurious ratios
    base_dir = 'results/'
    standard_dpi = 400
    sys.path.insert(0, '/usr/local/texlive/2023/bin/x86_64-darwin')

    
    # set the plotting style
    mpl.rcParams.update({
                'font.family': 'serif',
                'text.usetex': True,
                'pgf.rcfonts': False,
               
            })
    mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')



   

    if len(run_seed) == 1:
        run_seed_list = run_seed * len(methods) * len(datasets)
    else:
        run_seed_list = run_seed


    results_dict = dict.fromkeys(datasets, None)
    # create a dict to store the results
    for dataset in datasets:
        results_dict[dataset]  = dict.fromkeys(methods, None)
        for method in methods:
            results_dict[dataset][method] = dict.fromkeys(list_spurious_ratio, None)

            for spurious_ratio in list_spurious_ratio:
                results_dict[dataset][method][spurious_ratio] = dict.fromkeys(run_seed_list, None)

                for run_seed in run_seed_list:
                    results_dict[dataset][method][spurious_ratio][run_seed] = dict.fromkeys(['overall_acc', 'result_per_group_test'], None)



    for dataset in datasets:
        
        for method in methods:
            # create a dict for this method

        
            for run_seed_method in run_seed_list:

                print('Checking the sims for method: ', method, ' with run_seed: ', run_seed_method)

            
        

                for spurious_ratio in list_spurious_ratio:
                    # create a dict for this spurious ratio


                    # directory with results per sim
                    method_dir = '{}_{}_sims_{}_run_seed_{}_spurious_ratio_{}'.format(dataset, method, sims, run_seed_method, int(spurious_ratio*100))

                    # get all the .pkl files in this directory
                    list_files = os.listdir(base_dir + method_dir)

                    def name_ends_with_digit_and_pkl(filename):
                        return re.match(r'\w*\d+.pkl$', filename)
                

                    # create with all the files that  have .pkl at the end
                    list_files = [file for file in list_files if name_ends_with_digit_and_pkl(file)]
                    print('the list of files is: ', list_files)

                    list_overall_acc = []
                    list_acc_per_group = []
                    list_acc_per_group_train = []
                    list_d_c_method = []
                    list_d_m_method = []
                    list_V_c_method = []
                    list_V_m_method = []
                    list_X_train = []
                    list_X_val = []
                    list_y_c_train = []
                    list_y_m_train = []
                    list_y_m_val = []
                    list_weights = []

                    j = 0

                    # go over the .pkl files and load them
                    for file in list_files:   

                        print('the file is: {}'.format(file))             
                    


                        with open(base_dir + method_dir + '/' + file, 'rb') as f:
                            result_dict_for_sim = pickle.load(f)
                        
                        # get the overall accuracy
                        overall_acc = result_dict_for_sim['overall_acc_test']
                        list_overall_acc.append(overall_acc.item())

                        # get the accuracy per group
                        acc_per_group = result_dict_for_sim['result_per_group_test']
                        acc_per_group['sim'] = j + 1
                        list_acc_per_group.append(acc_per_group)

                        print('accuracy per group: ')
                        print(acc_per_group)

                        # get the accuracy per group for the training set
                        if 'result_per_group_train' in result_dict_for_sim.keys():
                            acc_per_group_train = result_dict_for_sim['result_per_group_train']
                            acc_per_group_train['sim'] = j + 1
                            list_acc_per_group_train.append(acc_per_group_train)

                        if 'd_c' in result_dict_for_sim.keys():
                            d_c_method = result_dict_for_sim['d_c']
                            list_d_c_method.append(d_c_method)

                        if 'd_m' in result_dict_for_sim.keys():
                            d_m_method = result_dict_for_sim['d_m']
                            list_d_m_method.append(d_m_method)

                        if 'V_c' in result_dict_for_sim.keys():
                            V_c_method = result_dict_for_sim['V_c']
                            list_V_c_method.append(V_c_method)
                        
                        if 'V_m' in result_dict_for_sim.keys():
                            V_m_method = result_dict_for_sim['V_m']
                            list_V_m_method.append(V_m_method)

                        if 'X_train' in result_dict_for_sim.keys():
                            X_train = result_dict_for_sim['X_train']
                            list_X_train.append(X_train)

                        if 'X_val' in result_dict_for_sim.keys():
                            X_val = result_dict_for_sim['X_val']
                            list_X_val.append(X_val)

                        if 'y_c_train' in result_dict_for_sim.keys():
                            y_c_train = result_dict_for_sim['y_c_train']
                            list_y_c_train.append(y_c_train)

                        if 'y_m_train' in result_dict_for_sim.keys():
                            y_m_train = result_dict_for_sim['y_m_train']
                            list_y_m_train.append(y_m_train)

                        if 'y_m_val' in result_dict_for_sim.keys():
                            y_m_val = result_dict_for_sim['y_m_val']
                            list_y_m_val.append(y_m_val)

                        if 'weights' in result_dict_for_sim.keys():
                            weights = result_dict_for_sim['weights']
                            results_dict[dataset][method][spurious_ratio][run_seed_method]['weights'] = weights
                            list_weights.append(weights)


                
                        if j == 0:
                            param_method_spurious_ratio = result_dict_for_sim['parameters']

                        j += 1

                    

                    

                    # combine list of overall acc into dataframe with two columns; sim and overall_acc
                    df_overall_acc = pd.DataFrame(list_overall_acc, columns=['overall_acc'])
                    df_overall_acc['spurious_ratio'] = spurious_ratio
                    results_dict[dataset][method][spurious_ratio][run_seed_method]['overall_acc'] = df_overall_acc

                    # combine list of acc per group into dataframe with three columns; sim, group, and acc
                    df_acc_per_group = pd.concat(list_acc_per_group)
                    df_acc_per_group['spurious_ratio'] = spurious_ratio
                    results_dict[dataset][method][spurious_ratio][run_seed_method]['result_per_group_test'] = df_acc_per_group

                
                    



                    # combine list of acc per group (train) into dataframe with three columns; sim, group, and acc
                    if 'result_per_group_train' in result_dict_for_sim.keys():
                        df_acc_per_group_train = pd.concat(list_acc_per_group_train)
                        df_acc_per_group_train['spurious_ratio'] = spurious_ratio
                        results_dict[dataset][method][spurious_ratio][run_seed_method]['result_per_group_train'] = df_acc_per_group_train


                    # create pd.Dataframe with d_c for the method
                    if 'd_c' in result_dict_for_sim.keys():
                        df_d_c_method = pd.DataFrame(list_d_c_method, columns=['d_c'])
                        df_d_c_method['spurious_ratio'] = spurious_ratio
                        df_d_c_method['sim'] = np.arange(1, len(list_d_c_method)+1)
                        df_d_c_method['method'] = method
                        results_dict[dataset][method][spurious_ratio][run_seed_method]['d_c'] = df_d_c_method

                    if 'd_m' in result_dict_for_sim.keys():
                        df_d_m_method = pd.DataFrame(list_d_m_method, columns=['d_m'])
                        df_d_m_method['spurious_ratio'] = spurious_ratio
                        df_d_m_method['sim'] = np.arange(1, len(list_d_m_method)+1)
                        df_d_m_method['method'] = method
                        results_dict[dataset][method][spurious_ratio][run_seed_method]['d_m'] = df_d_m_method

                    # save the parameters for this method and spurious ratio
                    results_dict[dataset][method][spurious_ratio][run_seed_method]['parameters'] = param_method_spurious_ratio

                    # save the V_c and V_m for this method and spurious ratio
                    if 'V_c' in result_dict_for_sim.keys():
                        results_dict[dataset][method][spurious_ratio][run_seed_method]['V_c'] = list_V_c_method
                    
                    if 'V_m' in result_dict_for_sim.keys():
                        results_dict[dataset][method][spurious_ratio][run_seed_method]['V_m'] = list_V_m_method


                    if 'X_train' in result_dict_for_sim.keys():
                        results_dict[dataset][method][spurious_ratio][run_seed_method]['X_train'] = list_X_train

                    if 'X_val' in result_dict_for_sim.keys():
                        results_dict[dataset][method][spurious_ratio][run_seed_method]['X_val'] = list_X_val

                    if 'y_c_train' in result_dict_for_sim.keys():
                        results_dict[dataset][method][spurious_ratio][run_seed_method]['y_c_train'] = list_y_c_train
                    
                    if 'y_m_train' in result_dict_for_sim.keys():
                        results_dict[dataset][method][spurious_ratio][run_seed_method]['y_m_train'] = list_y_m_train

                    if 'y_m_val' in result_dict_for_sim.keys():
                        results_dict[dataset][method][spurious_ratio][run_seed_method]['y_m_val'] = list_y_m_val


                    if 'weights' in result_dict_for_sim.keys():
                        results_dict[dataset][method][spurious_ratio][run_seed_method]['weights'] = list_weights
                

                
                # combine the results for all spurious ratios into one dataframe
                df_overall_acc_method = pd.concat([results_dict[dataset][method][spurious_ratio][run_seed_method]['overall_acc'] for spurious_ratio in list_spurious_ratio])
                df_acc_per_group_method = pd.concat([results_dict[dataset][method][spurious_ratio][run_seed_method]['result_per_group_test'] for spurious_ratio in list_spurious_ratio])
            
                results_dict[dataset][method][run_seed_method] = dict.fromkeys(['combined_overall_acc', 'combined_result_per_group_test', 'combined_result_per_group_train', 'combined_d_c', 'parameters'], None)
                results_dict[dataset][method][run_seed_method]['combined_overall_acc'] = df_overall_acc_method
                results_dict[dataset][method][run_seed_method]['combined_result_per_group_test'] = df_acc_per_group_method

                if 'result_per_group_train' in results_dict[dataset][method][spurious_ratio].keys():
                    df_acc_per_group_train_method = pd.concat([results_dict[dataset][method][spurious_ratio][run_seed_method]['result_per_group_train'] for spurious_ratio in list_spurious_ratio])
                    results_dict[dataset][method][run_seed_method]['combined_result_per_group_train'] = df_acc_per_group_train_method

                if 'd_c' in results_dict[dataset][method][spurious_ratio].keys():
                    df_d_c_method = pd.concat([results_dict[dataset][method][spurious_ratio][run_seed_method]['d_c'] for spurious_ratio in list_spurious_ratio])
                    results_dict[dataset][method][run_seed_method]['combined_d_c'] = df_d_c_method

                # these should all be the same across spurious ratios
                results_dict[dataset][method][run_seed_method]['parameters'] = param_method_spurious_ratio



    
    def create_score_per_spurious_ratio_shaded_plot(spurious_ratio, score, se_score, label, color, add_to_axis=False, ax=None, adj=100):

        if not add_to_axis:
            plt.plot(spurious_ratio, score, label=label, color=color, marker='.', markersize=2)
            plt.fill_between(spurious_ratio, adj*score-(adj*se_score*1.96), adj*score+(adj*se_score*1.96), color=color, alpha=0.2)
        else:
            ax.plot(spurious_ratio, score*adj, label=label, color=color, marker='.', markersize=2)
            ax.fill_between(spurious_ratio, adj*score-(adj*se_score*1.96), adj*score+(adj*se_score*1.96), color=color, alpha=0.2)

    def create_average_and_worst_group_per_spurious_ratio_shaded_plot(ax1, ax2, method, color, run_seed, label=None):

            if label is None:
                label = method
           
            ### First, overall accuracy
            # get the overall accuracy
            df_overall_acc_method = results_dict[dataset][method][run_seed]['combined_overall_acc']
            df_overall_acc_method['method'] = method


            # aggregate the overall accuracy per spurious ratio and method
            df_overall_acc_method_agg = df_overall_acc_method.groupby(['spurious_ratio', 'method']).agg({'overall_acc': ['mean', 'std']}).reset_index()
            df_overall_acc_method_agg.columns = ['spurious_ratio', 'method', 'mean_acc', 'std_acc']

            # turn the std into standard error
            df_overall_acc_method_agg['se_acc'] = df_overall_acc_method_agg['std_acc'] / np.sqrt(sims)

            # plot the accuracy
            create_score_per_spurious_ratio_shaded_plot(df_overall_acc_method_agg['spurious_ratio'].values, df_overall_acc_method_agg['mean_acc'].values, df_overall_acc_method_agg['se_acc'].values, label, color, add_to_axis=True, ax=ax1)

            ### Second, worst group accuracy
            # get the result per group for each simulation
            df_per_group_acc_method = results_dict[dataset][method][run_seed][wg_key]
            df_per_group_acc_method['method'] = method

            # replace the column name 'mean' with 'acc'
            df_per_group_acc_method = df_per_group_acc_method.rename(columns={'mean': 'acc'})
  
            # per spurious ratio, method, and simulation, select the worst group
            df_overall_acc_method_worst_group = df_per_group_acc_method.groupby(['spurious_ratio', 'method', 'sim'])['acc'].min().reset_index()
  
            # get the average and standard error of the worst group per spurious ratio and method
            df_overall_acc_method_worst_group_agg = df_overall_acc_method_worst_group.groupby(['spurious_ratio', 'method']).agg({'acc': ['mean', 'std']}).reset_index()
            df_overall_acc_method_worst_group_agg.columns = ['spurious_ratio', 'method', 'mean_wg_acc', 'std_wg_acc']

            # turn the std into standard error
            df_overall_acc_method_worst_group_agg['se_wg_acc'] = df_overall_acc_method_worst_group_agg['std_wg_acc'] / np.sqrt(sims)

            create_score_per_spurious_ratio_shaded_plot(df_overall_acc_method_worst_group_agg['spurious_ratio'].values, df_overall_acc_method_worst_group_agg['mean_wg_acc'].values, df_overall_acc_method_worst_group_agg['se_wg_acc'].values, label, color, add_to_axis=True, ax=ax2)
            
    # define colors for the methods for each plot
    colors = {'JSE': 'blue', 'INLP': 'red', 'ERM': 'orange', 'RLACE': 'purple', 'Group-weighted ERM': 'limegreen', 'ADV': 'green'}


    def illustrate_method_plot_Toy(ax_to_add, method, sim_index, vector_index, show_dgp_vectors, project_out_concept_vector, dataset_setting, legend, show,  color_concept_vec='red', color_main_vec='cyan', annotate_main_spurious_feature=False, annotate_corner=None, save_result=False):

            if method == 'ERM':


                W_ERM = results_dict[dataset][method][spurious_ratio]['weights'][sim_index].squeeze(0)
              

            else:
                # get the first concept vector from V_c, and first main-task vector from V_m
                V_c = results_dict[dataset][method][spurious_ratio]['V_c'][sim_index] # selecting for  sim
                v_c_1 = V_c[:, 0]

                # if there are two concept vectors, get the second one
                if V_c.shape[1] > 1:
                    v_c_2 = V_c[:, 1]

            
                # get the main-task vector if applicable
                if method == 'JSE':
                    V_m = results_dict[dataset][method][spurious_ratio]['V_m'][sim_index] # selecting for  sim
                    v_m_1 = V_m[:, 0]

                    if V_m.shape[1] > 1:
                        v_m_2 = V_m[:, 1]        

            # get the training data and labels
            X_train = results_dict[dataset][method][spurious_ratio]['X_train'][sim_index]
            X_val = results_dict[dataset][method][spurious_ratio]['X_val'][sim_index]
            y_c_train = results_dict[dataset][method][spurious_ratio]['y_c_train'][sim_index]
            y_m_train = results_dict[dataset][method][spurious_ratio]['y_m_train'][sim_index]
            y_m_val = results_dict[dataset][method][spurious_ratio]['y_m_val'][sim_index]


            # create the data object
            data_obj = Toy_Dataset()

          
            if method == 'INLP' or method == 'JSE':
                # if project_out_concept_vector, then remove the concept vector from the data
                if project_out_concept_vector:
                    
                    # if there are two concept vectors, put them together in a matrix
                    if vector_index == 1:
                        # put the first two concept vectors together in matrix
                        V_c = torch.stack([v_c_1, v_c_2], dim=1)
                        P_orth = torch.eye(20) - create_P(V_c)
                    else:
                        P_orth = torch.eye(20) - create_P(v_c_1.unsqueeze(1))


                    # project out the concept vector
                    X_train = torch.matmul(X_train, P_orth)
                    X_val = torch.matmul(X_val, P_orth)

                    # create the data object
                    data_obj.y_m_train = y_m_train
                    data_obj.y_m_val = y_m_val
                    data_obj.reset_X(X_train, X_val, batch_size=128, workers=0, shuffle=True, only_main=True, reset_X_objects=True)

                    # estimate v_m after projection
                    device = torch.device('cpu')
                    linear_model_after_projection = return_linear_model(20, 
                                        data_obj.main_loader,
                                        device,
                                        'SGD', 
                                        lr=0.1,
                                        epochs=50,
                                        weight_decay=0,
                                        early_stopping=True,
                                        bias=True
                                        )
                    w_m_est = linear_model_after_projection.linear_layer.weight.data.squeeze()
                    v_m_est_1 = w_m_est / torch.norm(w_m_est, p=2)

                    new_x_m_1 = torch.matmul(X_train, v_m_est_1)

                    # project the v_m_est from the X_train
                    P_orth_v_m = torch.eye(20) - create_P(v_m_est_1.unsqueeze(1))
                    X_train = torch.matmul(X_train, P_orth_v_m)

                    # create the data object              
                    data_obj.reset_X(X_train, X_val, batch_size=128, workers=0, shuffle=True, only_main=True, reset_X_objects=True)

                    # get the second most separating vector
                    linear_model_after_projection = return_linear_model(20, 
                                        data_obj.main_loader,
                                        device,
                                        'SGD', 
                                        lr=0.1,
                                        epochs=50,
                                        weight_decay=0,
                                        early_stopping=True,
                                        bias=True
                                        )
                    w_m_est = linear_model_after_projection.linear_layer.weight.data.squeeze()
                    v_m_est_2 = w_m_est / torch.norm(w_m_est, p=2)
                    new_x_m_2 = torch.matmul(X_train, v_m_est_2)

                    # define the two new features
                    X_1_plot = new_x_m_2
                    X_2_plot = new_x_m_1

                    # do not show the original vectors
                    show_dgp_vectors = False
                
                    # show the second concept vector
                    if vector_index == 0:
                        vector_index_to_show = 1
                    else:
                        vector_index_to_show = -1

                    # define the indices
                    X_1_index = 0
                    X_2_index = 1

                else:
                    X_1_index = 0
                    X_2_index = 1
                    X_1_plot = None
                    X_2_plot = None
                    vector_index_to_show = vector_index
            else:
                X_1_index = 0 
                X_2_index = 1
                X_1_plot = None
                X_2_plot = None
                
            data_obj.plot_groups(ax_to_add, y_m_train, y_c_train, X_train, 
                    colors= ['green', 'orange'], markers =['^', 'P'], alpha=0.5, s=15, 
                    groups = [1,2,3,4],  X_1_index = X_1_index, X_2_index = X_2_index, facecolor_type='fill', adendum_label='', legend=legend, X_1_plot=X_1_plot, X_2_plot=X_2_plot, edgecolor_type='fill')
            
            

            if method == 'JSE' or method == 'INLP':
                # get the decision boundary for first concept vector
                decision_boundary_v_c_1, concept_span = get_linear_combination(v_c_1, -4, 4, X_index=0)

                if V_c.shape[1] > 1:
                    decision_boundary_v_c_2, concept_span = get_linear_combination(v_c_2, -4, 4, X_index=0)


                if method == 'JSE':
                    decision_boundary_v_m_1, main_span = get_linear_combination(v_m_1, -4, 4, X_index=1)

                    if V_m.shape[1] > 1:
                        decision_boundary_v_m_2, main_span = get_linear_combination(v_m_2, -4, 4, X_index=1)
                    if vector_index_to_show == 0:
                        label = r'$\hat{\boldsymbol{v}}_{\mathrm{mt}, 1}$'
                        decision_boundary_to_show = decision_boundary_v_m_1
                        ax_to_add.plot(main_span, decision_boundary_to_show, color=color_main_vec, label=label, linestyle='--')

                    elif vector_index_to_show == 1 and V_c.shape[1] > 1:
                        label = r'$\hat{\boldsymbol{v}}_{\mathrm{mt}, 2}$'
                        decision_boundary_to_show = decision_boundary_v_m_2
                        ax_to_add.plot(main_span, decision_boundary_to_show, color=color_main_vec, label=label, linestyle='--')

                

                if vector_index_to_show == 0:
                    label = r'$\hat{\boldsymbol{v}}_{\mathrm{sp}, 1}$'
                    decision_boundary_to_show = decision_boundary_v_c_1
                    ax_to_add.plot(concept_span, decision_boundary_to_show, color=color_concept_vec, label=label, linestyle='--')
                elif vector_index_to_show == 1 and V_c.shape[1] > 1:
                    label = r'$\hat{\boldsymbol{v}}_{\mathrm{sp}, 2}$'
                    decision_boundary_to_show = decision_boundary_v_c_2
                    ax_to_add.plot(concept_span, decision_boundary_to_show, color=color_concept_vec, label=label, linestyle='--')

            
            min_axis = -4
            max_axis = 4
            ax_to_add.set_xlim(min_axis,max_axis)
            ax_to_add.set_ylim(min_axis,max_axis)

            if method == 'ERM':
                decision_boundary_ERM, span = get_linear_decision_boundary(W_ERM, xmin=min_axis, xmax=max_axis )
                decision_boundary_to_show = decision_boundary_ERM
                label = 'Group-weighted ERM'
                ax_to_add.plot(span, decision_boundary_to_show, color='limegreen', label= label, linestyle='--')
                

            



                
            
            if method == 'INLP' or method == 'JSE':
                if show_dgp_vectors:

                    zorder_arrow = 5
                    main_vec_label = r'${\boldsymbol{v}}_{\mathrm{mt}, 1}$'
                    spurious_vec_label = r'${\boldsymbol{v}}_{\mathrm{sp}, 1}$'


                    def make_legend_arrow(legend, orig_handle,
                                        xdescent, ydescent,
                                        width, height, fontsize):
                        p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
                        return p
                    if dataset_setting == 'default':
                        arrow_main_task = ax_to_add.arrow(0, 0, 0, 1, width=0.05, facecolor=color_main_vec, head_width=0.2, head_length=0.1, linestyle='-', label=main_vec_label, zorder=zorder_arrow,  edgecolor='black')
                        arrow_spurious = ax_to_add.arrow(0, 0, 1, 0, width=0.05, facecolor=color_concept_vec, head_width=0.2, head_length=0.1, linestyle='-', label=spurious_vec_label, zorder=zorder_arrow,  edgecolor='black')
                    
                    elif dataset_setting =='non_orthogonal':

                        # arrow that has 15 degrees angle with x axis
                        arrow_main_task = ax_to_add.arrow(0, 0, 0.2588, 0.9659, width=0.05, facecolor=color_main_vec, head_width=0.2, head_length=0.1, linestyle='-', label=main_vec_label, zorder=zorder_arrow,  edgecolor='black')
                        arrow_spurious = ax_to_add.arrow(0, 0, 1, 0, width=0.05, facecolor=color_concept_vec, head_width=0.2, head_length=0.1, linestyle='-', label=spurious_vec_label, zorder=zorder_arrow,  edgecolor='black')

            
                if legend:
                    legend = ax_to_add.legend(loc = 'lower right', )



                if show_dgp_vectors:

                    # Get the handles and labels from the existing legend
                    if legend:
                        existing_handles, existing_labels = legend.legendHandles, [t.get_text() for t in legend.get_texts()]

                        # remove the last two items from existing handles and labels
                        existing_handles = existing_handles[:-2]
                        existing_labels = existing_labels[:-2]

                        # Combine handles and labels
                        all_handles = existing_handles + [arrow_main_task, arrow_spurious]
                        all_labels = existing_labels + [obj.get_label() for obj in [arrow_main_task, arrow_spurious]]


                        # Get the existing handler map
                        existing_handler_map = legend.get_legend_handler_map()

                        # Create a new handler map based on the existing one
                        new_handler_map = existing_handler_map.copy()

                        def make_legend_arrow(legend, orig_handle,xdescent, ydescent, width, height, fontsize):
                            p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
                            return p

                        new_handler_map[type(arrow_main_task)] = HandlerPatch(patch_func=make_legend_arrow)

                        # remove existing legend, add new one
                        legend.remove()
                        legend = ax_to_add.legend(all_handles, all_labels, handler_map=new_handler_map)




            if annotate_main_spurious_feature:

                    # at the end of the X-axis, put the symbol: r$\boldsymbol{x}_{\mathrm{sp}}$
                    position_x_sp = 3
                    ax_to_add.annotate(r'$\boldsymbol{x}_{\mathrm{sp}}$', xy=(position_x_sp, 0), xytext=(position_x_sp, -0.75), fontsize=12)

                    # at the end of the Y-axis, put the symbol: r$\boldsymbol{x}_{\mathrm{mt}}$
                    position_x_mt = 3
                    ax_to_add.annotate(r'$\boldsymbol{x}_{\mathrm{mt}}$', xy=(0, position_x_mt), xytext=(-1.25, position_x_mt), fontsize=12)


            if annotate_corner is not None:
                ax_to_add.annotate(annotate_corner, xy=(-4, 4), xytext=(-3.75, 3.75), fontsize=16)

            if save_result: 

                file_tex ='plots_tables/illustrate_method_plot_method_{}_dataset_{}_sims_{}_spurious_ratio_{}_vector_index_{}_project_out_concept_vector_{}.tex'.format(method,  dataset, sims, int(100*spurious_ratio), vector_index, int(project_out_concept_vector))
                
                tikz_save(file_tex,
                axis_height = '\\figureheight',
                axis_width = '\\figurewidth'
                )
                
            if show:
                plt.show()

    if plot_type == 'accuracy_and_worst_group_plot_vision':

        rows = 2
        cols = 2
        fig, axes = plt.subplots(nrows=rows, ncols=cols)


        
        
        wg_key = 'combined_result_per_group_test'
        i = 0
        print('datasets are: ', datasets)


        for row in range(rows):
            

            
            if row == 0:
                dataset = 'Waterbirds'
                ax1 = axes[0, 0]
                ax2 = axes[0, 1]

            elif row == 1:
                dataset = 'celebA'
                ax3 = axes[1, 0]
                ax4 = axes[1, 1]
                
                # plot the accuracy
            for method in methods:

                
                # get the parameters for this method
                param_method = results_dict[dataset][method]['parameters']

                if group_weighted_ERM and method == 'ERM':
                    label = 'Group-weighted ERM'
                    color = colors[label]
                else:
                    label = None
                    color  = colors[method]
                    

                if dataset == 'Waterbirds':
                    create_average_and_worst_group_per_spurious_ratio_shaded_plot(ax1, ax2, method, color, label)
                elif dataset == 'celebA':
                    create_average_and_worst_group_per_spurious_ratio_shaded_plot(ax3, ax4, method, color, label)

                

    

   
        xlabel = r'$p_{\mathrm{train}}(y_{\mathrm{mt}} | y_{\mathrm{sp}})$ '
        ylabel_1 = r'Accuracy'
        ylabel_2 = r'Worst-Group Accuracy'



        ax1.legend(loc='lower left')
       

        ax3.set_xlabel(xlabel)
        ax4.set_xlabel(xlabel)

        ax1.set_title(ylabel_1)
        ax2.set_title(ylabel_2)

      

        ax3.set_xticks(list_spurious_ratio) 
        ax3.set_xticklabels(list_spurious_ratio)
        ax3.tick_params(axis='both', which='both', labelbottom=True)
        ax4.set_xticks(list_spurious_ratio)
        ax4.set_xticklabels(list_spurious_ratio)
        ax4.tick_params(axis='both', which='both', labelbottom=True)


        ax1.yaxis.set_major_formatter(PercentFormatter(decimals=0, xmax=100)) # 1 decimal 
        ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0, xmax=100)) # 1 decimal
        ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax3.yaxis.set_major_formatter(PercentFormatter(decimals=0, xmax=100)) # 1 decimal
        ax4.yaxis.set_major_formatter(PercentFormatter(decimals=0, xmax=100)) # 1 decimal
        ax3.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax4.yaxis.set_major_locator(plt.MaxNLocator(4))


        ax1.set_title('Accuracy               Waterbirds', x=0, y=0)
        ax3.set_title('Accuracy                CelebA', x=0, y=0)


        ax1.margins(x=0.025)
        ax2.margins(x=0.025)
        ax3.margins(x=0.025)
        ax4.margins(x=0.025)


       

        


        if save_result:

         
            plot_name = 'avg_worst_group_plot_index_{}_mode_{}_vision_sims_{}'.format(plot_index, mode, sims)
            
            file_tex = 'plots_tables/'+plot_name + '.tex'

      
            tikz_save(file_tex,
                 axis_height = '\\figureheight',
                 axis_width = '\\figurewidth'
                
                 )

            
        
        plt.show()

    elif plot_type == 'separability_plot':

        dataset = 'Toy'

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=False)
        ax1 = axes[0]
        ax2 = axes[1]


        # get the blue palette
        labels= [r'Adjustment in $\Delta$',  r'No adjustment in $\Delta$']
        colors = ['deepskyblue', 'orangered']


         # plot the accuracy
        i = 0
        wg_key = 'combined_result_per_group_test'
        
        for method in ['JSE']:
            ax = axes[i]
            j = 0

            for run_seed in run_seed_list:
                label = labels[j]
                color = colors[j]
                j += 1

                create_average_and_worst_group_per_spurious_ratio_shaded_plot(ax1, ax2, method,color, label=label, run_seed=run_seed)



                
        ax1.legend(loc='lower left', fontsize=5)
        ax2.legend(loc='lower left', fontsize=5)
        

        xlabel = r'$\rho$'
        ylabel_1 = r'Avg. Accuracy'
        ylabel_2 = r'Worst-Group Accuracy'
      

        ax2.set_xlabel(xlabel, fontsize=5)
        ax1.set_xlabel(xlabel, fontsize=5)

        ax1.set_xticks(list_spurious_ratio, fontsize=5) 
        ax1.set_xticklabels(list_spurious_ratio, fontsize=5)
        ax1.tick_params(axis='both', which='both', labelbottom=True)

        ax2.set_xticks(list_spurious_ratio, fontsize=5) 
        ax2.set_xticklabels(list_spurious_ratio, fontsize=5)
        ax2.tick_params(axis='both', which='both', labelbottom=True)

        ax1.set_title(ylabel_1)
        ax2.set_title(ylabel_2)

        ax1.yaxis.set_major_formatter(PercentFormatter(decimals=0, xmax=100)) # 1 decimal 
        ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0, xmax=100)) # 1 decimal
        ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax1.tick_params(axis='y', which='major', labelsize=5)
        ax2.tick_params(axis='y', which='major', labelsize=5)


        ax1.margins(x=0.025)
        ax2.margins(x=0.025)

        if save_result:
         
            plot_name = 'separability_plot_{}_mode_{}_dataset_{}_sims_{}'.format(plot_index, mode, dataset, sims)
            
            file_tex = 'plots_tables/'+plot_name + '.tex'

      
            tikz_save(file_tex,
                axis_height = '\\figureheight',
                axis_width = '\\figurewidth'
                
                )
        

        plt.show()



    elif plot_type == 'sample_size_plot':

        dataset = 'Toy'

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=False)
        ax1 = axes[0]
        ax2 = axes[1]

      
        labels = [r'$n = 500$', r'$n = 1000$', r'$n = 5000$', r'$n = 10000$']
        unique_values_run_seed_list = list(np.unique(np.array(run_seed_list)))
        
        # get the blue palette
        palette_JSE = sns.color_palette("Blues", len(unique_values_run_seed_list))
        palette_INLP = sns.color_palette("Reds", len(unique_values_run_seed_list))



         # plot the accuracy
        i = 0
        
        for method in methods:
            ax = axes[i]
            j = 0

           

            for run_seed in unique_values_run_seed_list:
                if method == 'JSE':
                    color = palette_JSE[j]
                elif method == 'INLP':
                    color = palette_INLP[j]

                label = labels[j]


                
                # get the parameters for this method
                print(results_dict[dataset][method].keys())
                print('run_seed is: ', run_seed)
                param_method = results_dict[dataset][method][run_seed]['parameters']

               
                    


                ### First, overall accuracy
                # get the overall accuracy
                df_overall_acc_method = results_dict[dataset][method][run_seed]['combined_overall_acc']
                df_overall_acc_method['method'] = method


                # aggregate the overall accuracy per spurious ratio and method
                df_overall_acc_method_agg = df_overall_acc_method.groupby(['spurious_ratio', 'method']).agg({'overall_acc': ['mean', 'std']}).reset_index()
                df_overall_acc_method_agg.columns = ['spurious_ratio', 'method', 'mean_acc', 'std_acc']

                # turn the std into standard error
                df_overall_acc_method_agg['se_acc'] = df_overall_acc_method_agg['std_acc'] / np.sqrt(sims)

                # plot the accuracy
                create_score_per_spurious_ratio_shaded_plot(df_overall_acc_method_agg['spurious_ratio'].values, df_overall_acc_method_agg['mean_acc'].values, df_overall_acc_method_agg['se_acc'].values, label, color, add_to_axis=True, ax=ax)

                j+=1
            i+=1  
                
        ax1.legend(loc='lower left', fontsize=5)
        ax2.legend(loc='lower left', fontsize=5)
        

        xlabel = r'$\rho$'
        ylabel_1 = r'JSE - Accuracy '
        ylabel_2 = r'INLP - Accuracy '
      

        ax2.set_xlabel(xlabel, fontsize=5)
        ax1.set_xlabel(xlabel, fontsize=5)

        ax1.set_xticks(list_spurious_ratio, fontsize=5) 
        ax1.set_xticklabels(list_spurious_ratio, fontsize=5)
        ax1.tick_params(axis='both', which='both', labelbottom=True)

        ax2.set_xticks(list_spurious_ratio, fontsize=5) 
        ax2.set_xticklabels(list_spurious_ratio, fontsize=5)
        ax2.tick_params(axis='both', which='both', labelbottom=True)

        ax1.set_title(ylabel_1)
        ax2.set_title(ylabel_2)

        ax1.yaxis.set_major_formatter(PercentFormatter(decimals=0, xmax=100)) # 1 decimal 
        ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0, xmax=100)) # 1 decimal
        ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax1.tick_params(axis='y', which='major', labelsize=5)
        ax2.tick_params(axis='y', which='major', labelsize=5)


        ax1.margins(x=0.025)
        ax2.margins(x=0.025)

        if save_result:
         
            plot_name = 'sample_size_plot_{}_mode_{}_dataset_{}_sims_{}'.format(plot_index, mode, dataset, sims)
            
            file_tex = 'plots_tables/'+plot_name + '.tex'

      
            tikz_save(file_tex,
                axis_height = '\\figureheight',
                axis_width = '\\figurewidth'
                
                )
        

        plt.show()




    elif plot_type == 'accuracy_and_worst_group_plot':



        # create two subplots, vertically
        figsize = (7, 4.5)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=figsize)

    
        wg_key = 'combined_result_per_group_test'



        # plot the accuracy
        for method in methods:

             
            # get the parameters for this method
            param_method = results_dict[dataset][method]['parameters']

            if group_weighted_ERM and method == 'ERM':
                label = 'Group-weighted ERM'
                color = colors[label]
            else:
                label = None
                color  = colors[method]
                


            create_average_and_worst_group_per_spurious_ratio_shaded_plot(ax1, ax2, method,color, label=label)
            
            
        ax1.legend(loc='lower left', fontsize=5)
        

        if dataset == 'Toy':
            xlabel = r'$\rho_{\mathrm{sp}, \mathrm{mt}}$'
            ylabel_1 = r'Accuracy '
            ylabel_2 = r'Worst-Group Accuracy '
        else:
            xlabel = r'$p_{\mathrm{train}}(y_{\mathrm{mt}} | y_{\mathrm{sp}})$ '
            ylabel_1 = r'Accuracy'
            ylabel_2 = r'Worst-Group Accuracy'


        ax2.set_xlabel(xlabel, fontsize=5)
        ax1.set_xlabel(xlabel, fontsize=5)

        ax1.set_xticks(list_spurious_ratio, fontsize=5) 
        ax1.set_xticklabels(list_spurious_ratio, fontsize=5)
        ax1.tick_params(axis='both', which='both', labelbottom=True)

        ax1.set_title(ylabel_1)
        ax2.set_title(ylabel_2)


        ax1.yaxis.set_major_formatter(PercentFormatter(decimals=0, xmax=100)) # 1 decimal 
        ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0, xmax=100)) # 1 decimal
        ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax1.tick_params(axis='y', which='major', labelsize=5)
        ax2.tick_params(axis='y', which='major', labelsize=5)


        ax1.margins(x=0.025)
        ax2.margins(x=0.025)




        if save_result:
         
            plot_name = 'avg_worst_group_plot_index_{}_mode_{}_dataset_{}_sims_{}'.format(plot_index, mode, dataset, sims)
            
            file_tex = 'plots_tables/'+plot_name + '.tex'

      
            tikz_save(file_tex,
                axis_height = '\\figureheight',
                axis_width = '\\figurewidth'
                
                )
        
        plt.show()


    elif plot_type == 'accuracy_plot':  

        if save_result:
            mpl.use('pgf')    
        for method in methods:
            param_method = results_dict[dataset][method]['parameters']


            df_overall_acc_method = results_dict[dataset][method]['combined_overall_acc']
            df_overall_acc_method['method'] = method

            df_overall_acc_method_agg = df_overall_acc_method.groupby(['spurious_ratio', 'method']).agg({'overall_acc': ['mean', 'std']}).reset_index()
            df_overall_acc_method_agg.columns = ['spurious_ratio', 'method', 'mean_acc', 'std_acc']
            print(df_overall_acc_method_agg)

            # turn the std into standard error
            df_overall_acc_method_agg['se_acc'] = df_overall_acc_method_agg['std_acc'] / np.sqrt(sims)

            color = colors[method]
            create_score_per_spurious_ratio_shaded_plot(df_overall_acc_method_agg['spurious_ratio'].values, df_overall_acc_method_agg['mean_acc'].values, df_overall_acc_method_agg['se_acc'].values, method, color)
        
        
        plt.legend()
        plt.xticks(list_spurious_ratio)

        if dataset == 'Toy':
            xlabel = r'$\rho_{\mathrm{sp}, \mathrm{mt}}$ in train data'
            ylabel = r'Accuracy for test data'
        else:
            xlabel = r'$p(y_{\mathrm{mt}} | y_{\mathrm{sp}})$ in train data '
            ylabel = r'Accuracy for test data'

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if save_result:
            plot_name = 'accuracy_plot_index_{}_mode_{}_dataset_{}_sims_{}'.format(plot_index, mode, dataset, sims)
            file_pdf = 'plots_tables/'+plot_name + '.pdf'
            file_tex = 'plots_tables/'+plot_name + '.pgf'
            plt.savefig(file_pdf, dpi=standard_dpi)
            plt.savefig(file_tex, dpi=standard_dpi)
        plt.show()

    elif plot_type == 'worst_group_plot':

        if mode =='train':
            key = 'combined_result_per_group_train'
        elif mode == 'test':
            key = 'combined_result_per_group_test'
        
       
        if save_result:
            mpl.use('pgf')    
        for method in methods:
            param_method = results_dict[dataset][method]['parameters']


            # get the result per group for each simulation
            df_per_group_acc_method = results_dict[dataset][method][key]
            df_per_group_acc_method['method'] = method

            # replace the column name 'mean' with 'acc'
            df_per_group_acc_method = df_per_group_acc_method.rename(columns={'mean': 'acc'})
  

            # per spurious ratio, method, and simulation, select the worst group
            df_overall_acc_method_worst_group = df_per_group_acc_method.groupby(['spurious_ratio', 'method', 'sim'])['acc'].min().reset_index()
  
            # get the average and standard error of the worst group per spurious ratio and method
            df_overall_acc_method_worst_group_agg = df_overall_acc_method_worst_group.groupby(['spurious_ratio', 'method']).agg({'acc': ['mean', 'std']}).reset_index()
            df_overall_acc_method_worst_group_agg.columns = ['spurious_ratio', 'method', 'mean_wg_acc', 'std_wg_acc']

            # turn the std into standard error
            df_overall_acc_method_worst_group_agg['se_wg_acc'] = df_overall_acc_method_worst_group_agg['std_wg_acc'] / np.sqrt(sims)

            color = colors[method]
            create_score_per_spurious_ratio_shaded_plot(df_overall_acc_method_worst_group_agg['spurious_ratio'].values, df_overall_acc_method_worst_group_agg['mean_wg_acc'].values, df_overall_acc_method_worst_group_agg['se_wg_acc'].values, method, color)
            
        plt.legend()
        plt.xticks(list_spurious_ratio)

        if dataset == 'Toy':
            xlabel = r'$\rho_{c, m}$ in train data'
        
        else:
            xlabel = r'$p(y_m  | y_c)$ in train data '

        ylabel = r'Worst-Group Accuracy for test data'

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)



        if save_result:

            plot_name = 'worst_group_plot_index_{}_mode_{}_dataset_{}_sims_{}'.format(plot_index, mode, dataset, sims)
            file_pdf = 'plots_tables/'+plot_name + '.pdf'
            file_tex = 'plots_tables/'+plot_name + '.pgf'
            plt.savefig(file_pdf, dpi=standard_dpi)
            plt.savefig(file_tex, dpi=standard_dpi)

        plt.show()
    


    elif plot_type == 'dimensionality_subspaces':


        # the plot is; for a given method, x-axis is the spurious ratio, and y-axis is a histogram of the dimensionality of the subspaces
        if len(methods) > 1:
            print('Can only plot dimensionality of subspaces for one method at a time')
            sys.exit()

        

     

        if mode =='concept':
            key = 'd_c'
            df_d_method = df_d_c_method
        elif mode =='main':
            key = 'd_m'
            df_d_method = df_d_m_method



        list_df_per_spurious_ratio = []
        for spurious_ratio in list_spurious_ratio:
            
            # get the results for the given spurious ratio
            df_d_method_spurious_ratio = df_d_method[df_d_method['spurious_ratio'] == spurious_ratio]

            # get the dimensionality of the subspaces for the given spurious ratio
            d_method_spurious_ratio = df_d_method_spurious_ratio[key].values

            # get the unique values of the dimensionality of the subspaces, sort these from small to large
            unique_d_method_spurious_ratio = np.unique(d_method_spurious_ratio)
            unique_d_method_spurious_ratio = np.sort(unique_d_method_spurious_ratio)

            # get per unique value the count of the dimensionality of the subspaces
            count_d_method_spurious_ratio = np.zeros(len(unique_d_method_spurious_ratio))
            for i, d in enumerate(unique_d_method_spurious_ratio):
                count_d_method_spurious_ratio[i] = np.sum(d_method_spurious_ratio == d)
            
            # sort these counts such that it aligns with the order in unique_d_c_method_spurious_ratio
            count_d_method_spurious_ratio = count_d_method_spurious_ratio[np.argsort(unique_d_method_spurious_ratio)]

            # create dataframe with three columns; spurious ratio, the unique values of d_c, and the count per unique value
            df_d_method_spurious_ratio = pd.DataFrame({'spurious_ratio': spurious_ratio, key: unique_d_method_spurious_ratio, 'count': count_d_method_spurious_ratio})
            list_df_per_spurious_ratio.append(df_d_method_spurious_ratio)
        
        df_d_method_all_spurious_ratios = pd.concat(list_df_per_spurious_ratio)

        pivot_df = df_d_method_all_spurious_ratios.pivot(index='spurious_ratio', columns=key, values='count')

        print('pivot_df: {}'.format(pivot_df))
        fig, ax = plt.subplots()

            
        width = 0.2
        x = np.arange(len(pivot_df.index))

        # Adjust the x-axis positions for grouped bars
        x_grouped = [pos + (width * 1.5) * (i - 1) for i, pos in enumerate(x)]

        num_bars = len(pivot_df.columns)
        all_x_grouped = []
      
        for i, dc_val in enumerate(pivot_df.columns):
            idx_not_nan = ~np.isnan(pivot_df[dc_val].values)
            x_grouped_not_nan = list(np.array(x_grouped)[idx_not_nan])
            all_x_grouped.append(x_grouped_not_nan)

            


            bar = ax.bar(x_grouped, pivot_df[dc_val], width=width,  color=colors[method])
            x_grouped = [pos + width for pos in x_grouped]
            ax.bar_label(bar,labels=[dc_val]*len(x_grouped), label_type='center', fmt='%.0f', padding=3, color='white', size=6)

        # Set the xticks to the middle of each group


        

        print('X grouped', x_grouped)

        # Add text annotations using ax.bar_label
        i = 0
        j = 0
        xticks = [0]*len(pivot_df.index)

        # flatten the list of lists
        all_x_grouped = [item for sublist in all_x_grouped for item in sublist]

        # sor the list of lists
        all_x_grouped = np.sort(all_x_grouped)
        print('all_x_grouped: {}'.format(all_x_grouped))


        for sp_ratio in pivot_df.index:

            # select the values for d_c for which the values are not nan given the spurious ratio
            values = pivot_df.loc[sp_ratio].values

            # which of the values are not nan?
            not_nan = ~np.isnan(values)
            
            # how many bars?
            num_bars = np.sum(not_nan)
            i = i + num_bars 

            print('index of final bar: {}'.format(i))
            print('num_bars: {}'.format(num_bars))


            # where is the final bar put?
            pos_final_bar = all_x_grouped[i-1]
            print('pos_final_bar: {}'.format(pos_final_bar))

            ax.axvline(pos_final_bar + width, color='black', linestyle='dashed')

           
            pos_xtick = pos_final_bar - (width * (num_bars-1)/2) 
            xticks[j]  = pos_xtick


            j += 1

        
        ax.set_xticks(xticks)
        ax.set_xticklabels(pivot_df.index)

      



        # Set labels and title
        ax.set_xlabel(r'$p(y_m = y | y_c = y)$')
        ax.set_ylabel('Frequency')

        plt.tight_layout()
        if save_result:
            plot_name = 'dimensionality_subspace_plot_index_{}_method_{}_dataset_{}_sims_{}.png'.format(plot_index, method, dataset, sims)
            plt.savefig('plots_tables/'+plot_name, dpi=standard_dpi)
        plt.show()

    elif plot_type =='avg_worst_group_table':
        if mode =='train':
            key = 'combined_result_per_group_train'
        elif mode == 'test':
            key = 'combined_result_per_group_test'
            
        key_overall_acc = 'overall_acc'


        list_df_per_method = []

         # select the accuracy per group (train)
        for method in methods:
                
                # go over each spurious ratio
                list_overall_acc_method = []
                list_overall_acc_method_se = []
                for spurious_ratio in list_spurious_ratio:
                
                    # get the overall accuracy for each simulation
                    df_overall_acc = results_dict[dataset][method][spurious_ratio][run_seed][key_overall_acc]

                    # calculate the average and se for the overall accuracy
                    avg_overall_acc_method = df_overall_acc['overall_acc'].mean()
                    std_overall_acc_method = df_overall_acc['overall_acc'].std()
                    se_overall_acc_method = std_overall_acc_method/np.sqrt(sims)

                    # append the average and se to the list, per spurious ratio
                    list_overall_acc_method.append(avg_overall_acc_method)
                    list_overall_acc_method_se.append(se_overall_acc_method)



                # get the result per group for each simulation
                param_method = results_dict[dataset][method][run_seed]['parameters']


                print('Parameters for method {} are: {}'.format(method, param_method))

                # get the result per group for each simulation
                df_per_group_acc_method = results_dict[dataset][method][run_seed][key]
                df_per_group_acc_method['method'] = method

                # replace the column name 'mean' with 'acc'
                df_per_group_acc_method = df_per_group_acc_method.rename(columns={'mean': 'acc'}).reset_index()

                # create the group column based on main/concept columns - 1 is 0/0, 2 is 0/1, 3 is 1/0, 4 is 1/1
                data_obj = Dataset_obj()
                df_per_group_acc_method['group'] = data_obj.get_groups(df_per_group_acc_method['main'].values, df_per_group_acc_method['concept'].values)

                # select the worst group for each simulation
                df_group_acc = df_per_group_acc_method[['group', 'method', 'spurious_ratio', 'sim', 'acc']].groupby(['method', 'spurious_ratio', 'sim']).agg({'acc': 'min'}).reset_index()

                # select the average of the worst groups accuracy, per sim
                df_group_acc = df_group_acc.groupby(['method', 'spurious_ratio']).agg({'acc': ['mean', 'std']}).reset_index()

                # change the last column from acc to worst_group_acc
                df_group_acc.columns = ['method', 'spurious_ratio', 'worst_group_acc_mean', 'worst_group_acc_std']
                df_group_acc['worst_group_acc_se'] = df_group_acc['worst_group_acc_std'] / np.sqrt(sims)

                # add to the df of worst group acc
                df_group_acc['overall_acc_mean'] = list_overall_acc_method
                df_group_acc['overall_acc_se'] = list_overall_acc_method_se

                # create a new column (string) with worst-group accuracy, and in brackets the standard error. Set everything in percentages
                df_group_acc['worst_group_acc_mean'] = df_group_acc['worst_group_acc_mean'] * 100
                df_group_acc['worst_group_acc_se'] = df_group_acc['worst_group_acc_se'] * 100
                df_group_acc['worst_group_acc_str'] = df_group_acc.apply(lambda row: '{:.2f} ({:.2f})'.format(row['worst_group_acc_mean'], row['worst_group_acc_se']), axis=1)

                
                # create a new column (string) with average overall accuracy, and in brackets the standard error. Set everything in percentages
                df_group_acc['overall_acc_mean'] = df_group_acc['overall_acc_mean'] * 100
                df_group_acc['overall_acc_se'] = df_group_acc['overall_acc_se'] * 100
                df_group_acc['overall_acc_str'] = df_group_acc.apply(lambda row: '{:.2f} ({:.2f})'.format(row['overall_acc_mean'], row['overall_acc_se']), axis=1)

                df_worst_group_acc_agg = df_group_acc[['method','spurious_ratio', 'worst_group_acc_str']]
                df_overall_acc_agg = df_group_acc[['method','spurious_ratio', 'overall_acc_str']]

                #  create two new columns - accuracy type (overall, worst group), and value, containing the accuracy for that type
                df_worst_group_acc_agg['accuracy_type'] = 'worst_group'
                df_overall_acc_agg['accuracy_type'] = 'overall'

                # concatenate the two dataframes
                df_worst_group_acc_agg = df_worst_group_acc_agg.rename(columns={'worst_group_acc_str': 'accuracy_value'})
                df_overall_acc_agg = df_overall_acc_agg.rename(columns={'overall_acc_str': 'accuracy_value'})
                df_acc_agg = pd.concat([df_worst_group_acc_agg, df_overall_acc_agg])

                # pivot the dataframe - put the spurious ratio in the columns, the methods /groups in the rows
                df_acc_agg_selected = df_acc_agg.pivot_table(index=['method', 'accuracy_type' ], columns=['spurious_ratio'], values='accuracy_value', aggfunc='first').reset_index()
                list_df_per_method.append(df_acc_agg_selected)



    

        # concatenate the dataframes
        df_per_group_acc_all = pd.concat(list_df_per_method)
        print(df_per_group_acc_all)

        # write to csv
        if save_result:
            csv_name = 'df_avg_worst_group_acc_{}_mode_{}_dataset_{}_sims_{}.csv'.format(plot_index, mode, dataset, sims)
            df_per_group_acc_all.to_csv('plots_tables/' + csv_name, index=False)
            print("SAVED")


    elif plot_type == 'accuracy_per_group_table':

        if mode =='train':
            key = 'combined_result_per_group_train'
        elif mode == 'test':
            key = 'combined_result_per_group_test'
        
        list_df_per_method = []

        # select the accuracy per group (train)
        for method in methods:
                param_method = results_dict[dataset][method][run_seed]['parameters']



                print('Parameters for method {} are: {}'.format(method, param_method))

                # get the result per group for each simulation
                df_per_group_acc_method = results_dict[dataset][method][run_seed][key]
                df_per_group_acc_method['method'] = method

                # replace the column name 'mean' with 'acc'
                df_per_group_acc_method = df_per_group_acc_method.rename(columns={'mean': 'acc'}).reset_index()

                # create the group column based on main/concept columns - 1 is 0/0, 2 is 0/1, 3 is 1/0, 4 is 1/1
                data_obj = Dataset_obj()
                df_per_group_acc_method['group'] = data_obj.get_groups(df_per_group_acc_method['main'].values, df_per_group_acc_method['concept'].values)

                # get the average per group, main/concept, and spurious ratio, for method
                df_per_group_acc_method_agg = df_per_group_acc_method.groupby(['group', 'main', 'concept', 'spurious_ratio']).agg({'acc': ['mean', 'std']}).reset_index()

                # rename the columns
                df_per_group_acc_method_agg.columns = ['group', 'main', 'concept', 'spurious_ratio', 'acc', 'std']

                # add the standard error
                df_per_group_acc_method_agg['se'] = df_per_group_acc_method_agg['std'] / np.sqrt(sims)

                # sort based on spurious ratio, then the group
                df_per_group_acc_method_agg = df_per_group_acc_method_agg.sort_values(by=['spurious_ratio', 'group'])

                # add the method column
                df_per_group_acc_method_agg['method'] = method

                # create a new column (string) with average accuracy, and in brackets the standard error. Set everything in percentages
                df_per_group_acc_method_agg['acc'] = df_per_group_acc_method_agg['acc'] * 100
                df_per_group_acc_method_agg['se'] = df_per_group_acc_method_agg['se'] * 100
                df_per_group_acc_method_agg['acc_str'] = df_per_group_acc_method_agg.apply(lambda row: '{:.2f} ({:.2f})'.format(row['acc'], row['se']), axis=1)

                
                df_per_group_acc_method_agg_selected = df_per_group_acc_method_agg[['method', 'group', 'main', 'concept', 'spurious_ratio', 'acc_str']]

                # pivot the dataframe - put the spurious ratio in the columns, the methods /groups in the rows
                df_per_group_acc_method_agg_selected = df_per_group_acc_method_agg_selected.pivot(index=['method', 'group', 'main', 'concept'], columns='spurious_ratio', values='acc_str').reset_index()
                list_df_per_method.append(df_per_group_acc_method_agg_selected)

        # concatenate the dataframes
        df_per_group_acc_all = pd.concat(list_df_per_method)
        print(df_per_group_acc_all)

        # write to csv
        if save_result:
            csv_name = 'df_acc_per_group_index_{}_mode_{}_dataset_{}_sims_{}.csv'.format(plot_index, mode, dataset, sims)
            df_per_group_acc_all.to_csv('plots_tables/' + csv_name, index=False)
            print("SAVED")

    elif plot_type == 'illustrate_method_plot_orthogonality_assumption':
        

        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True )    

        #1 The first subplot
        illustrate_method_plot_Toy(ax_to_add= axes[0], method='INLP', sim_index=0, vector_index=0, show_dgp_vectors=False, project_out_concept_vector=False, dataset_setting='default', legend=False, show=False, color_concept_vec='red', color_main_vec='blue', annotate_main_spurious_feature=True, save_result=False)
        #2 The second subplot

        illustrate_method_plot_Toy(ax_to_add= axes[1], method='JSE', sim_index=0, vector_index=0, show_dgp_vectors=False, project_out_concept_vector=False, dataset_setting='default', legend=False, show=False, color_concept_vec='red', color_main_vec='blue', annotate_main_spurious_feature=True, save_result=False)
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

        # remove the last item from existing handles and labels
        lines = lines[:-1]
        labels = labels[:-1]

        # select lines, labels for the legend
        line_vec_sp = lines[4]
        line_vec_mt = lines[-1]
        lines =  lines[0:4] + [line_vec_sp, line_vec_mt]

        label_vec_sp = labels[4]
        label_vec_mt = labels[-1]
        labels =  labels[0:4] + [label_vec_sp, label_vec_mt]


        plt.legend(lines, labels, loc='lower right', ncol=1, bbox_to_anchor=(1.95, 0.35))
        axes[0].set_title('INLP')
        axes[1].set_title('JSE')
        plt.tight_layout()

        if save_result:

            
            plot_name = 'illustrate_method_plot_orthogonality_assumption_index_{}_mode_{}_dataset_{}_sims_{}.tex'.format(plot_index, mode, dataset, sims)

            tikz_save('plots_tables/'+plot_name,
                axis_height = '\\figureheight',
                axis_width = '\\figurewidth'
            )

        
        plt.show()


    elif plot_type == 'illustrate_method_plot_toy_dataset':


        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True )     # 6 axes, returned as a 2-d array



        axes[0, 0].set_ylabel('INLP')
        axes[1, 0].set_ylabel('JSE')
        axes[0, 0].set_title(r'Rank: $d$')
        axes[0, 1].set_title(r'Rank: $d-1$')
        axes[0, 2].set_title(r'Rank: $d-2$')

        #1 The first subplot
        illustrate_method_plot_Toy(ax_to_add= axes[0, 0], method='INLP', sim_index=0, vector_index=0, show_dgp_vectors=False, project_out_concept_vector=False, dataset_setting='default', legend=False, show=False, color_concept_vec='red', color_main_vec='blue', annotate_main_spurious_feature=True, save_result=False, annotate_corner='(A)')#'INLP', 0, 0, True, False, 'default', False, legend=False, show=False)

        #2 The second subplot
        illustrate_method_plot_Toy(ax_to_add=axes[1, 0], method='JSE', sim_index=0, vector_index=0, show_dgp_vectors=False, project_out_concept_vector=False, dataset_setting='default', legend=False, show=False, color_concept_vec='red', color_main_vec='blue', annotate_main_spurious_feature=True, save_result=False, annotate_corner='(D)')#'INLP', 0, 0, True, False, 'default', False, legend=False, show=False)


        #  3 The third subplot
        illustrate_method_plot_Toy(ax_to_add=axes[0, 1], method='INLP', sim_index=0, vector_index=0, show_dgp_vectors=False, project_out_concept_vector=True, dataset_setting='default', legend=False, show=False, color_concept_vec='purple', save_result=False,  annotate_corner='(B)')#'INLP', 0, 0, True, False, 'default', False, legend=False, show=False)

        # 4 The fourth subplot
        illustrate_method_plot_Toy(ax_to_add=axes[1, 1], method='JSE', sim_index=0, vector_index=0, show_dgp_vectors=False, project_out_concept_vector=True, dataset_setting='default', legend=False, show=False, save_result=False,  annotate_corner='(E)')

        # 5 The fifth subplot
        illustrate_method_plot_Toy(ax_to_add=axes[0, 2], method='INLP', sim_index=0, vector_index=1, show_dgp_vectors=False, project_out_concept_vector=True, dataset_setting='default', legend=False, show=False, save_result=False,  annotate_corner='(C)')


        # delete the last subplot
        fig.delaxes(axes[1, 2])
        
        # get all 
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        
        # select the first 5 elements
        lines_plot = lines[0:5]
        labels_plot= labels[0:5]

        # select the last element for the second subplot
        lines_second_subplot, labels_second_subplot =  fig.axes[3].get_legend_handles_labels()
        lines_plot.append(lines_second_subplot[-2])
        labels_plot.append(labels_second_subplot[-2])

        # select the last element for the third subplot
        lines_third_subplot, labels_third_subplot =  fig.axes[1].get_legend_handles_labels()
        lines_plot.append(lines_third_subplot[-1])
        labels_plot.append(labels_third_subplot[-1])


        
        plt.legend(lines_plot, labels_plot, loc='lower right', bbox_to_anchor=(2.22, 0.1), frameon=False)

        if save_result:
                plot_name = 'illustrate_method_plot_toy_dataset_index_{}_mode_{}_dataset_{}_sims_{}.tex'.format(plot_index, mode, dataset, sims)
    
                tikz_save('plots_tables/'+plot_name,
                    axis_height = '\\figureheight',
                    axis_width = '\\figurewidth'
                )

        
        plt.show()

    elif plot_type == 'illustrate_method_plot':
        

        dataset_setting = 'default'
        fig, ax = plt.subplots(1, 1)
        illustrate_method_plot_Toy(ax, method=method, sim_index=0, vector_index=0, show_dgp_vectors=False, project_out_concept_vector=False, dataset_setting=dataset_setting, legend=True, show=False, color_concept_vec='red', color_main_vec='blue', annotate_main_spurious_feature=True, save_result=False, annotate_corner=None)
        # get all 
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        
        plt.legend(lines, labels, loc='lower right', bbox_to_anchor=(2.25, 0.1), frameon=False)


        if save_result:
                print('saving result')
                plot_name = 'illustrate_method_plot_setting_{}_{}_mode_{}_dataset_{}_sims_{}.tex'.format(dataset_setting, plot_index, mode, dataset, sims)
    
                tikz_save('plots_tables/'+plot_name,
                    axis_height = '\\figureheight',
                    axis_width = '\\figurewidth'
                )

        
        plt.show()

    if save_result:
            print('For index, the result was created with the following paramers:')
            print('run_seed: {}'.format(run_seed))
            print('mode: {}'.format(mode))
            print('dataset: {}'.format(dataset))
            print('sims: {}'.format(sims))

            print('The parameters for each method were:')
            for method in methods:
                print('Method: {}'.format(method))
                print('Parameters: {}'.format(results_dict[dataset][method][run_seed]['parameters']))
            
            
            
            

               






        
       

        
        










if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use")
    parser.add_argument("--methods",  help="methods to compare")
    parser.add_argument("--spurious_ratio",  help="show for which spurious ratios?")
    parser.add_argument("--run_seed",  default='1971', help="seed for the run")
    parser.add_argument("--sims", type=int, default=5, help="spurious ratio")
    parser.add_argument("--plot_type", type=str, default="accuracy_test", help="what type of plot to make?")
    parser.add_argument('--mode', type=str, default='test', help='specifies further for plot')
    parser.add_argument('--save_result', type=str,  help='save the results?')
    parser.add_argument('--plot_index', type=int, default=0, help='index of the plot')
    parser.add_argument('--vector_index', type=int, default=0, help='index of the vector')
    parser.add_argument('--project_out_concept_vector', type=str, default='True', help='project out the concept vector?')
    parser.add_argument('--group_weighted_ERM', type=str, default='False', help='show group weighted ERM?')
   
    args = parser.parse_args()
    dict_arguments = vars(args)

    dataset = Convert(dict_arguments["dataset"], str)
    spurious_ratio = Convert(dict_arguments["spurious_ratio"], float)
    methods = Convert(dict_arguments["methods"], str)
    run_seed = Convert(dict_arguments["run_seed"], int)
    sims = dict_arguments["sims"]
    plot_type = dict_arguments["plot_type"]
    mode = dict_arguments["mode"]
    save_result = str_to_bool(dict_arguments["save_result"])
    plot_index = dict_arguments["plot_index"]

    vector_index = dict_arguments["vector_index"]
    project_out_concept_vector = str_to_bool(dict_arguments["project_out_concept_vector"])
    group_weighted_ERM = str_to_bool(dict_arguments["group_weighted_ERM"])

    print('vector_index: {}'.format(vector_index))
    print('project_out_concept_vector: {}'.format(project_out_concept_vector))



    main(dataset, methods,spurious_ratio, run_seed, sims, plot_type, mode, save_result, plot_index, vector_index, project_out_concept_vector, group_weighted_ERM)

