

optimizer_info = {
    'All': {
        'tol': 1e-3,
        'patience': 5
    },

    'standard_ERM_settings': {
        'Waterbirds': {
            'weight_decay': 0.01,
            'lr': 0.01,
            'batch_size': 128
        },
        'celebA': { 
            'weight_decay': 0.01,
            'lr': 0.01, 
            'batch_size': 128 
        },
        'multiNLI': {
            'weight_decay': 1, 
            'lr': 0.01,
            'batch_size': 128
        },
        'Toy': {
            'weight_decay': 0.00,
            'lr': 0.1,
            'batch_size': 128

        }
    },

    'standard_ERM_settings_GW':  {
        'Waterbirds': {
            'weight_decay': 0.01,
            'lr': 0.01,
            'batch_size': 128
        },
        'celebA': { 
            'weight_decay': 0.01,
            'lr': 0.01, 
            'batch_size': 128 
        },
        'multiNLI': {
            'weight_decay': 0.1, 
            'lr': 0.01,
            'batch_size': 128
        },
        'Toy': {
            'weight_decay': 0.00,
            'lr': 0.1,
            'batch_size': 128

        }
    },

    'standard_JSE_settings': {
        'Waterbirds': {
            'weight_decay': 0.001, 
            'lr': 0.001,
            'batch_size': 128
        },
        'celebA': {
            'weight_decay': 0.01, 
            'lr': 0.001, 
            'batch_size': 128  
        },
        'multiNLI': {
             'weight_decay': 0.01,
               'lr': 0.01, 
               'batch_size': 128
        },
        'Toy': {
            'weight_decay': 0.0,
            'lr': 0.01,
            'batch_size': 128
        }
    },

    'standard_INLP_settings':{
       'Toy': {
            'weight_decay': 0.0,
            'lr': 0.1,
            'batch_size': 128
        },
        'Waterbirds': {
            'weight_decay': 0.001,
            'lr': 0.1,
            'batch_size': 128
        },
        'celebA': { 
            'weight_decay': 0.01,
            'lr': 0.001,
            'batch_size': 128
        },
        'multiNLI': {
            'weight_decay': 0.001, 
            'lr': 0.1,
            'batch_size': 128
        }

    
        
    },

    'standard_RLACE_settings': {
        'Toy': {
            'weight_decay': 0.0, 
            'lr': 0.1,
            'batch_size': 128
        },
        'Waterbirds': {
            'weight_decay': 0.001, 
            'lr': 0.1,
            'batch_size': 128
        },
        'celebA': { 
            'weight_decay': 0.01, 
            'lr': 0.001,
            'batch_size': 128
        },
        'multiNLI': {
            'weight_decay': 0.001,
            'lr': 0.1,
            'batch_size': 512

        }
    },
    
}



data_info = {
    'All': {
        'train_split': 0.8
    },
    'celebA': {
        'default': {
            'set_sample':False,
            'main_task_name': 'Young',
            'concept_name': 'Female',
             'adversarial':False,
            'combine_sets': False
        },
        'sampled_data': {
            'set_sample':True,
            'p_m':0.5,
            'train_size': 4500,
            'val_size': 2000,
            'test_size': 2000,
            'main_task_name': 'Blond_Hair',
            'concept_name': 'Female',
             'adversarial':False,
            'combine_sets': True
        },
        'sampled_data_adv': {
            'set_sample':False,
            'p_m':0.5,
            'train_size': 4500,
            'val_size': 2000,
            'test_size': 2000,
            'main_task_name': 'Blond_Hair',
            'concept_name': 'Female',
            'adversarial':True,
            'settings': 'adversarial_param',
            'early_stopping': True

            },
       
    },
    'Toy': {
    'default': {
        'n': 2000,
        'd': 20,
        'gamma_c': 3,
        'gamma_m': 3,
        'rho_c': 0.0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':0
      

        },

    'sample_size_1000':{
        'n': 1000,
        'd': 20,
        'gamma_c': 3,
        'gamma_m': 3,
        'rho_c': 0.0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':0
    },

    'sample_size_500':{
        'n': 500,
        'd': 20,
        'gamma_c': 3,
        'gamma_m': 3,
        'rho_c': 0.0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':0
    },

    'sample_size_5000':{
        'n': 5000,
        'd': 20,
        'gamma_c': 3,
        'gamma_m': 3,
        'rho_c': 0.0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':0
    },

    'sample_size_10000':{
        'n': 10000,
        'd': 20,
        'gamma_c': 3,
        'gamma_m': 3,
        'rho_c': 0.0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':0
    },

    'non_orthogonal': {'n': 2000,
        'd': 20,
        'gamma_c': 3,
        'gamma_m': 3,
        'rho_c': 0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':15
      

        },
    'different_separability': {'n': 2000,
        'd': 20,
        'gamma_c': 6,
        'gamma_m': 2,
        'angle': 90,
        'rho_c': 0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':0

        },
    },

    'Waterbirds':{
        'default': {
            'corr_test':'50',
            'image_type_train_val': 'combined',
            'image_type_test': 'combined',
            'model_name': 'resnet50',
             'balance_main': False,
            'adversarial':False


    },
        'adv': {
            'corr_test':'50',
            'image_type_train_val': 'combined',
            'image_type_test': 'combined',
            'model_name': 'resnet50',
             'balance_main': True,
             'adversarial':True,
             
        },
        'balanced': {
            'corr_test':'50',
            'image_type_train_val': 'combined',
            'image_type_test': 'combined',
            'model_name': 'resnet50',
            'balance_main': True,
            'adversarial':False,

        },

    },
    'multiNLI':{ 
        'default': {
            'finetune_param_type': 'adversarial_param',
            'embedding_type': 'CLS',
            'finetuned_BERT': 1,
            'train_size': 50000,
            'val_size': 5000,
            'binary_task': True,
            'dropout': 0.1,
            'early_stopping': True,
            'finetune_mode': 'CLS',
            'finetune_seed':1971,
            'spurious_ratio_train': None
            
        },
        'from_50_50_sample': {
             'finetune_param_type': 'adversarial_param',
            'embedding_type': 'CLS',
            'finetuned_BERT': 1,
            'train_size': 50000,
            'val_size': 5000,
            'binary_task': True,
            'dropout': 0.1,
            'early_stopping': True,
            'finetune_mode': 'CLS',
            'finetune_seed':1971,
            'spurious_ratio_train': 0.5,
            
        },
    } 
}
