import os
import sys
import numpy as np
import random
import string
import pandas as pd
from JSE.data import *
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import pickle

def main():
    ################ Paths and other configs - Set these #################################

    data_dir = 'raw_input/MNLI_GLUE/original'
    save_dir = 'raw_input/multinli_bert_features_punctuated'

    # If 'preset', use the official train/val/test MultiNLI split
    # If 'random', randomly split 50%/20%/30% of the data to train/val/test

    ######################################################################################

    ### Helper functions

    # tokenize the sentence
    def tokenize(s):
        s = s.translate(str.maketrans('', '', string.punctuation))
        s = s.lower()
        s = s.split(' ')
        return s

    ### Read in data and assign train/val/test splits
    train_df = pd.read_json(
        os.path.join(
            data_dir,
            'multinli_1.0_train.jsonl'),
        lines=True)

    val_df = pd.read_json(
        os.path.join(
            data_dir,
            'multinli_1.0_dev_matched.jsonl'),
        lines=True)

    test_df = pd.read_json(
        os.path.join(
            data_dir,
            'multinli_1.0_dev_mismatched.jsonl'),
        lines=True)

    # assign 0, 1, 2, denoting train, val, test
    split_dict = {
        'train': 0,
        'val': 1,
        'test': 2
    }

    # set 20% for validation, 30% for test
    val_frac = 0.2
    test_frac = 0.3

    # create dataframe with all data
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # shuffle the data, assign splits
    n = len(df)
    n_val = int(val_frac * n)
    n_test = int(test_frac * n)
    n_train = n - n_val - n_test
    splits = np.array([split_dict['train']] * n_train + [split_dict['val']] * n_val + [split_dict['test']] * n_test)
    np.random.shuffle(splits)
    df['split'] = splits

    ### Assign labels
    df = df.loc[df['gold_label'] != '-', :]
    print(f'Total number of examples: {len(df)}')
    for k, v in split_dict.items():
        print(k, np.mean(df['split'] == v))

    label_dict = {
        'contradiction': 0,
        'entailment': 1,
        'neutral': 2
    }
    for k, v in label_dict.items():
        idx = df.loc[:, 'gold_label'] == k
        df.loc[idx, 'gold_label'] = v

   

    # load the original data
    multiNLI_obj = multiNLI_Dataset()
    multiNLI_obj.load_main_dataset('raw_input/multinli_bert_features', 'processed', turn_binary=True, )


    # get bert uncased tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    

    def encoder(tokenizer, sentence1, sentence2):
                
        
        result = tokenizer.encode_plus(sentence1,
                                        sentence2, 
                                        add_special_tokens=True,
                                        max_length=128,
                                        pad_to_max_length=True,
                                        return_attention_mask=True,
                                        return_token_type_ids=True,
                                        return_tensors='pt'
                                        )
        return result
    
    n_samples = len(df.index)

    list_input_ids = [None]*n_samples
    list_attention_masks = [None]*n_samples
    list_token_type_ids = [None]*n_samples
    list_punctuation = [0]*n_samples

    perc_with_punctuation = 0.5
    counter = 0
  
    for i in range(n_samples):

        if i % 1000 == 0:
            print('at sample: ', i, ' out of ', n_samples, ' samples')

        sentence_1_i = df['sentence1'][i:i+1].values[0]
        sentence_2_i = df['sentence2'][i:i+1].values[0]

        # in 50% of the cases, add '!!' to the second sentence
        if random.random() < perc_with_punctuation:
            if sentence_2_i[-1] == '.':
                sentence_2_i = sentence_2_i[:-1] 
            sentence_2_i = sentence_2_i + '!!'
            list_punctuation[i] = 1

            

        result = encoder(tokenizer, sentence_1_i, sentence_2_i)
        list_input_ids[i] = result['input_ids']

        if result['input_ids'][0][-1] != 0:
            if result['input_ids'][0][-2] != 99:
                # due to truncation, !! is removed
                list_punctuation[i] = 0
                counter += 1
        
        list_attention_masks[i] = result['attention_mask']
        list_token_type_ids[i] = result['token_type_ids']
    
    # convert to tensors
    print(list_input_ids)
    input_ids = torch.cat(list_input_ids, dim=0)
    attention_masks = torch.cat(list_attention_masks, dim=0)
    token_type_ids = torch.cat(list_token_type_ids, dim=0)
    punctuation = torch.tensor(list_punctuation)

    df['sentence2_has_punctuation'] = punctuation.numpy().astype(int)


    df = df[['gold_label', 'sentence2_has_punctuation', 'split']]
    df.to_csv(os.path.join(save_dir, f'metadata_multiNLI_punctuated.csv'))

    # create dict, save that dict
    data_dict = {
        'input_ids': input_ids,
        'attention_masks': attention_masks,
        'token_type_ids': token_type_ids,
        'punctuation': punctuation
    }

    print('counter: ', counter)
    with open(os.path.join(save_dir, f'multiNLI_punctuated.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)
       

        

 
if __name__ == "__main__":


    main()