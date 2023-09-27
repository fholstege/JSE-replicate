
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import sys
import math
import sklearn
from twokenize import simpleTokenize
import argparse
from JSE.helpers import *
from JSE.data import *
import re
import pickle
import gc

def concat_last_four_embeddings(token_embeddings):
    # Stores the token vectors, with shape [128 x 3,072]
    token_vecs_cat = []

    # `token_embeddings` is a [128 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:
        
        # `token` is a [12 x 768] tensor

        # Concatenate the vectors (that is, append them together) from the last 
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        
        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)

    return token_vecs_cat


def has_none_string(lst):
    for element in lst:
        if not isinstance(element, str):
            return True
    return False



    


def main(finetuned_BERT, settings, train_size, val_size, spurious_ratio_train,spurious_ratio_sample, embedding_type, binary_task, dropout,  device,use_punctuation,  seed_list,early_stopping=True, finetune_mode='CLS' ):



    for seed in seed_list:

        # define the model
        bert = BertModel.from_pretrained(  "bert-base-uncased") 
        
        bert.to(device)

        # load the finetuned BERT model
        if finetuned_BERT:

            finetuned_file_name = 'finetuned_models/bert_model_finetuned_MNLI_settings_{}_train_size_{}_val_size_{}_spurious_ratio_{}_binary_{}_dropout_{}_early_stopping_{}_finetune_mode_{}_seed_{}'.format( settings, train_size, val_size, int(100*spurious_ratio_train), int(binary_task), int(100*dropout), int(early_stopping),finetune_mode, seed)

            if use_punctuation:
                finetuned_file_name +=  '_punctuation'
            finetuned_file_name += '.pt'


            parameters = torch.load(finetuned_file_name)

            
            bert.load_state_dict(parameters)
            print('Loaded finetuned BERT model: {}'.format(finetuned_file_name))

        # set to device
        bert.to(device)


        def create_embeddings(bert, input_ids, token_type_ids, attention_masks, embedding_type='pooled_output'):


            # get ids, token type, attention mask for subsequent BERT encoding. Turn to tensors
            input_ids =input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_masks = attention_masks.to(device)

            # encode the batch of tokens, and select the embedding type
            bert_output = bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks, output_hidden_states=True)

            if embedding_type == 'pooled_output':
                # get the pooled output
                embedding = bert_output['pooler_output']

            elif embedding_type == 'avg_last_hidden_state':
                
                # get the last hidden state
                token_embeddings = bert_output.last_hidden_state
            
                # Calculate the average of all token embeddings.
                embedding = token_embeddings.mean(dim=1)

            
            elif embedding_type == 'max_last_hidden_state':

                # get the last hidden state
                token_embeddings = bert_output.last_hidden_state
            
                # Calculate the max of all token embeddings.
                embedding = token_embeddings.max(dim=0)
            
            elif embedding_type == 'CLS':

                # get the last hidden state
                embedding = bert_output.last_hidden_state[:,0,:].detach()


            elif embedding_type =='last_4_avg':

                # get the last 4 hidden states
                last_4_layers = bert_output.hidden_states[-4:]
                
                last_layer_avg = torch.mean(last_4_layers[-1], dim=1)
                second_to_last_layer_avg = torch.mean(last_4_layers[-2], dim=1)
                third_to_last_layer_avg = torch.mean(last_4_layers[-3], dim=1)
                fourth_to_last_layer_avg = torch.mean(last_4_layers[-4], dim=1)

                # concatenate the last 4 layers
                embedding = torch.cat((last_layer_avg, second_to_last_layer_avg, third_to_last_layer_avg, fourth_to_last_layer_avg), dim=1)
                            

            elif embedding_type == 'last_4_max':

                # get the last 4 hidden states
                last_4_layers = bert_output.hidden_states[-4:]
                
                last_layer_avg = torch.max(last_4_layers[-1], dim=1)[0]
                second_to_last_layer_avg = torch.max(last_4_layers[-2], dim=1)[0]
                third_to_last_layer_avg = torch.max(last_4_layers[-3], dim=1)[0]
                fourth_to_last_layer_avg = torch.max(last_4_layers[-4], dim=1)[0]

                # concatenate the last 4 layers
                embedding = torch.cat((last_layer_avg, second_to_last_layer_avg, third_to_last_layer_avg, fourth_to_last_layer_avg), dim=1)
                            
            



            return embedding


        def get_embeddings(bert, input_ids,token_type_ids, attention_masks, embedding_type='pooled_output', set_embedding_to_cpu=True):
            """
            Adapted from Rafvogel (2022)

            """
            
            all_embeddings = []
            bert.eval()
            total = input_ids.shape[0]
            with torch.no_grad():
                
                # define the size of each batch. Not of importance when BERT is frozen
                batch_size = 4
                
                # go over each text in the dataset, in batches
                for i in range(0, total, batch_size):
                        print('at index: {}/{}'.format(i,total))

                        # select a batch of texts
                        batch_input_ids = input_ids[i:i+batch_size]
                        batch_token_type_ids = token_type_ids[i:i+batch_size]
                        batch_attention_masks = attention_masks[i:i+batch_size]

                        # get the embedding 
                        embedding = create_embeddings(bert, batch_input_ids, batch_token_type_ids, batch_attention_masks, embedding_type=embedding_type)
                        
                        if set_embedding_to_cpu:
                            embedding = embedding.cpu()

                        # add the embedding to the list of embeddings
                        all_embeddings.append(embedding.detach())

                        del embedding
                        del batch_input_ids
                        del batch_token_type_ids
                        del batch_attention_masks

                        if i % 1000 == 0:
                            gc.collect()
    

                # if there are remaining texts, get the embeddings
                embeddings = torch.cat(all_embeddings)
        
            print('embeddings shape: ', embeddings.shape)
            return embeddings

        

        # load the data
        multiNLI_obj = multiNLI_Dataset()

        if not use_punctuation:
            multiNLI_obj.load_main_dataset('raw_input/multinli_bert_features', 'processed', turn_binary=binary_task)
        else:
            multiNLI_obj.load_main_dataset('raw_input/multinli_bert_features_punctuated', 'raw_input/multinli_bert_features_punctuated', turn_binary=binary_task, punctuation=True)
    

        # get the embeddings
        input_ids_train, input_ids_val, input_ids_test, input_masks_train, input_masks_val, input_masks_test, segment_ids_train, segment_ids_val, segment_ids_test, y_m_train, y_m_val, y_m_test, y_c_train, y_c_val, y_c_test= multiNLI_obj.return_train_val_test_data()


        if train_size =='all':

            train_embeddings = get_embeddings(bert, input_ids_train, segment_ids_train, input_masks_train, embedding_type=embedding_type )
            val_embeddings = get_embeddings(bert, input_ids_val, segment_ids_val, input_masks_val, embedding_type=embedding_type)
            test_embeddings = get_embeddings(bert, input_ids_test, segment_ids_test, input_masks_test, embedding_type=embedding_type)
        else:


            if finetuned_BERT:

                if spurious_ratio_sample != spurious_ratio_train:
                    print('The model is trained on spurious ratio: {}, but sample from {} '.format(spurious_ratio_train, spurious_ratio_sample))
                

                filename = 'finetuned_models/sample_ids_MNLI_settings_{}_train_size_{}_val_size_{}_spurious_ratio_{}_binary_{}_dropout_{}_early_stopping_{}_finetune_mode_{}_seed_{}'.format( settings, train_size, val_size, int(100*spurious_ratio_sample), int(binary_task),int(100*dropout), int(early_stopping), finetune_mode, seed)
                if use_punctuation:
                    filename += '_punctuation'
                filename += '.pkl'  



                # load the pickle with the dictionary containing sample ids
                with open(filename, 'rb') as handle:
                    sample_dict = pickle.load(handle)

                # get the sample ids
                sample_train_ids = sample_dict['train_sample']
                sample_val_ids = sample_dict['val_sample']
                sample_test_ids = sample_dict['test_sample']

                print('train size: {}'.format(len(sample_train_ids)))
                print('val size: {}'.format(len(sample_val_ids)))
                print('test size: {}'.format(len(sample_test_ids)))
            

            else:

                multiNLI_obj.set_sample_subset(train_size, val_size, val_size,  spurious_ratio_sample, True, not binary_task, seed)
                sample_train_ids = multiNLI_obj.sample_train_ids
                sample_val_ids = multiNLI_obj.sample_val_ids
                sample_test_ids = multiNLI_obj.sample_test_ids


            # get the embeddings for the train and validation set
            input_ids_train_sample = input_ids_train[sample_train_ids,]
            attention_masks_train_sample = input_masks_train[sample_train_ids,]
            segment_ids_train_sample = segment_ids_train[sample_train_ids,]

            print('input ids size: {}'.format(input_ids_train_sample.shape[0]))
            print('attention masks size: {}'.format(attention_masks_train_sample.shape[0]))
            print('segment ids size: {}'.format(segment_ids_train_sample.shape[0]))
            
        
            input_ids_val_sample = input_ids_val[sample_val_ids,]
            attention_masks_val_sample = input_masks_val[sample_val_ids,]
            segment_ids_val_sample = segment_ids_val[sample_val_ids,]

            if binary_task:
                y_m_train = y_m_train[sample_train_ids]
                y_c_train = y_c_train[sample_train_ids]
                y_m_val = y_m_val[sample_val_ids]
                y_c_val = y_c_val[sample_val_ids]
                y_m_test = y_m_test[sample_test_ids]
                y_c_test = y_c_test[sample_test_ids]
            else:
                y_m_train = y_m_train[sample_train_ids,]
                y_c_train = y_c_train[sample_train_ids,]
                y_m_val = y_m_val[sample_val_ids,]
                y_c_val = y_c_val[sample_val_ids,]
                y_m_test = y_m_test[sample_test_ids,]
                y_c_test = y_c_test[sample_test_ids,]

            print('y_m_train size: {}'.format(y_m_train.shape[0]))
            print('y_c_train size: {}'.format(y_c_train.shape[0]))

            print('Division of y_m/y_c in loaded data')
            print('n of samples where y_m = 1: {}'.format(torch.sum(y_m_train)))
            print('n of samples where y_m = 0: {}'.format(torch.sum(y_m_train==0)))
            print('n of samples where y_m = 1, y_c = 1: {}'.format(torch.sum((y_m_train==1) & (y_c_train==1))))
            print('n of samples where y_m = 1, y_c = 0: {}'.format(torch.sum((y_m_train==1) & (y_c_train==0))))
            print('n of samples where y_m = 0, y_c = 1: {}'.format(torch.sum((y_m_train==0) & (y_c_train==1))))
            print('n of samples where y_m = 0, y_c = 0: {}'.format(torch.sum((y_m_train==0) & (y_c_train==0))))

            # get the embeddings for the test set
            input_ids_test_sample = input_ids_test[sample_test_ids,]
            attention_masks_test_sample = input_masks_test[sample_test_ids,]
            segment_ids_test_sample = segment_ids_test[sample_test_ids,]
    
            train_embeddings = get_embeddings(bert, input_ids_train_sample, segment_ids_train_sample, attention_masks_train_sample, embedding_type=embedding_type)
            val_embeddings = get_embeddings(bert, input_ids_val_sample, segment_ids_val_sample, attention_masks_val_sample, embedding_type=embedding_type)
            test_embeddings = get_embeddings(bert, input_ids_test_sample, segment_ids_test_sample, attention_masks_test_sample, embedding_type=embedding_type)



        data_dict = {'train': {'embeddings': train_embeddings, 'y_m': y_m_train, 'y_c': y_c_train}, 'val': {'embeddings': val_embeddings, 'y_m': y_m_val, 'y_c': y_c_val}, 'test': {'embeddings': test_embeddings, 'y_m': y_m_test, 'y_c': y_c_test}}

        # save the data_dict as a pickle
        filename = 'MNLI_data_settings_{}_embedding_type_{}_finetuned_{}_train_size_{}_val_size_{}_spurious_ratio_train_{}_spurious_ratio_sample_{}_binary_{}_dropout_{}_early_stopping_{}_finetune_mode_{}_seed_{}'.format( settings, embedding_type, int(finetuned_BERT), train_size, val_size, int(100*spurious_ratio_train), int(100*spurious_ratio_sample), int(binary_task), int(100*dropout), int(early_stopping), finetune_mode, seed)

        if use_punctuation:
            filename += '_punctuation'
        filename += '.pkl'

        with open(filename, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        
        print('Saved data_dict to {}'.format(filename)) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--finetuned_BERT')
    parser.add_argument('--settings')
    parser.add_argument('--train_size')
    parser.add_argument('--val_size')
    parser.add_argument('--embedding_type')
    parser.add_argument('--finetuned_BERT_dir')
    parser.add_argument('--spurious_ratio_train')
    parser.add_argument('--spurious_ratio_sample')
    parser.add_argument('--binary_task')
    parser.add_argument('--dropout')
    parser.add_argument('--use_punctuation')
    parser.add_argument('--seed')


    args = parser.parse_args()
    dict_arguments = vars(args)
    finetuned_BERT = str_to_bool(dict_arguments['finetuned_BERT'])
    finetuned_BERT_dir = (dict_arguments['finetuned_BERT_dir'])
    embedding_type = dict_arguments['embedding_type']

    settings = dict_arguments['settings']
    train_size = (dict_arguments['train_size'])
    val_size = (dict_arguments['val_size'])

    if not train_size == 'all':
        train_size = int(train_size)
    if not val_size == 'all':
        val_size = int(val_size)


    binary_task = str_to_bool(dict_arguments['binary_task'])
    dropout = (dict_arguments['dropout'])

    if dropout is not None:
        dropout = float(dropout)
    spurious_ratio_train = float(dict_arguments['spurious_ratio_train'])
    spurious_ratio_sample = float(dict_arguments['spurious_ratio_sample'])
    use_punctuation = str_to_bool(dict_arguments['use_punctuation'])
    seed = Convert(dict_arguments['seed'], int)









    main(finetuned_BERT, settings, train_size, val_size, spurious_ratio_train, spurious_ratio_sample, embedding_type, binary_task, dropout,  'mps' ,use_punctuation,  seed, )

