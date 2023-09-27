# Removing Spurious Concepts Via Joint Subspace Estimation


This folder contains the code that was used for the submission 'Removing Spurious Concepts Via Joint Subspace Estimation' for ICLR 2024. 


In order to replicate results from the paper, the following steps need to be taken. 

1. It is recommended to start a new virtual environment before reproducing code
2. Install the required libraries, specified in the requirements.txt file, via  pip install -r requirements.txt 
3. In order to run some of the files, it is necessary to install JSE as a local library (in order to ensure correct imports). This can be done by going to the JSE folder and run pip install -e . 

This folder does not contain all the datasets used due to size issues. One can acquire these as follows:

* The Waterbirds dataset is made up of the places dataset ([link](http://places.csail.mit.edu)) and Caltech-UCSD Birds-200-2011 (CUB) dataset ([link](https://www.vision.caltech.edu/datasets/cub_200_2011/)). The celebA dataset can be found [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and the multiNLI dataset [here](https://gluebenchmark.com/tasks). 
* Save the raw datasets in the following folders: 
    -  Waterbirds: in 'raw_input', put the birds and segmentations under the respective folders, and the images from the places dataset under the 'water_land' folder
    -  CelebA: under the folder 'raw_input' put the images in the 'images' folder
    -  multiNLI: under the folder 'raw_input' in the folder put them in 'multinli_bert_features'

* Run respective files in order to create the datasets, and finetune models in order to create the embeddings. 
    -  Waterbirds: run create_waterbird_images.py, create_waterbird_embeddings.py, and if desired finetune_waterbirds.py
    -  CelebA: run create_waterbird_images.py, create_waterbird_embeddings.py, and if desired finetune_celebA.py
    -  multiNLI: run create_multiNLI_metadata.py, create_multiNLI_data_punctuation, create_multiNLI_embeddings.py, and if desired finetune_multiNLI.py


There is a short python notebook that gives an example for how to code for the Toy dataset and implement JSE under 'Example_of_implementation.ipynb'. 
We also added a brief notebook that shows how to use various terminal calls to generate results from the paper

Below is a description of the remaining files in the JSE folder. 
* **create_plots.py**: contains functions to create plots for the paper
* **data.py**: contains classes that perform data cleaning and prepare it for training
* **generate_all_results.py or generate_result_sim.py**: both files that generate results for a set number of simulations. The former can take multiple methods and will always use the basic settings defined in settings.py. 
* **generate_result_.....py**: generates results for a particular method. Note that ADV requires that there already exist embeddings from models finetuned with adversarial removal.
* **helpers.py**: generic helper functions for remainder of the code
* **models.py**: defines model classes. 'joint_model' is the one used by JSE. 
* **RLACE.py**: code from Rafvogel (2022) for RLACE
* **select_param.py**: defines methods for selecting parameters for ERM/INLP/JSE
* **settings.py**: contains basic settings, chosen after hyper-parameter selection procedure specified in the paper
* **training.py**: defines training functions. 'train_JSE' contains implementation of JSE algorithm, and 'train_INLP' the implementation for INLP
