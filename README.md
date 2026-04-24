# Robust Ranking
## Installation:
Recommended to use a virtual environment
Inside the virtual environment

git clone <repository>
Add the data folder
Add the models_local folder to keep all the models locally
Run pip install -r requirements.txt

---

## Training/ Evaluation
Run the script generic_training.py 
For further settings see parameters.py

The evaluation scores are computed on the fly during training

### Run Encoder models (E5 and Biencoder)
Choose a dataset of choice (ace2004, aquaint, iitb-fix, kore50, msnbc, n3reuters128, n3rss500, spotlight)
Replace the aida dataset with your choice --dataset aida

For E5: python generic_training.py --dataset aida --found_model e5 --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0
For biencoder: python generic_training.py --dataset aida --found_model biencoder --type_optimization all_encoder_layers --learning_rate 3e-8 --gpu_id 0

### Run decoder models (Llama and Qwen)
The way to run the decoder models is either using a hyperparameter search (run_lbw_hyperparameter_search.py) or running a specific configuration of the decoder model.

*FOR HYPERPARAMETER SEARCH (Llama models):*

Llama models tested:
1. meta-llama/Llama-3.2-1B
2. meta-llama/Llama-3.2-3B
3. meta-llama/Llama-3.1-8B

Choose a dataset of choice (ace2004, aquaint, iitb-fix, kore50, msnbc, n3reuters128, n3rss500, spotlight)
Change line 32: "--dataset", "spotlight"

It is recommended to download the models locally from huggingface and change the line 40 DEFAULT_MODEL_ID = <model_location>

Keep these for the llama models:
Line 33: "--found_model", "llama_decoder",
Line 34: "--type_optimization", "all_encoder_layers_llama_decoder",

Run the file: python3 -u run_lbw_hyperparameter_search.py --gpu_id 0
For Background process: setsid python3 -u run_lbw_hyperparameter_search.py --gpu_id 0 > <log_file_name>.log 2>&1 &
For viewing the Background process: tail -f <log_file_name>.log

*FOR HYPERPARAMETER SEARCH (Qwen3 models):*

Qwen3 models tested:
1. Qwen/Qwen3-Embedding-0.6B
2. Qwen/Qwen3-Embedding-4B
3. Qwen/Qwen3-Embedding-8B

Choose a dataset of choice (ace2004, aquaint, iitb-fix, kore50, msnbc, n3reuters128, n3rss500, spotlight)
Change line 32: "--dataset", "spotlight"

It is recommended to download the models locally from huggingface and change the line 40 DEFAULT_MODEL_ID = <model_location>

Keep these for the llama models:
Line 33: "--found_model", "qwen3_decoder",
Line 34: "--type_optimization", "all_encoder_layers_qwen3_decoder",

Run the file: python3 -u run_lbw_hyperparameter_search.py --gpu_id 0
For Background process: setsid python3 -u run_lbw_hyperparameter_search.py --gpu_id 0 > <log_file_name>.log 2>&1 &
For viewing the Background process: tail -f <log_file_name>.log
(If there is a lora adapter trained): setsid python3 -u run_lbw_hyperparameter_search.py --gpu_id 0 --lora_adapter_path <Lora_adapter_saved_folder> --lora_epoch <Lora_adapter_saved_epoch_folder> > <log_file_name>.log 2>&1 &

*FOR SPECIFIC CONFIGURATION (Llama models):*

Make changes: 
1. --model_id to include the folder where the local model is saved
2. Any further configuration changes for testing

python3 generic_training.py \
    --dataset ace2004 \
    --found_model llama_decoder \
    --type_optimization all_encoder_layers_llama_decoder \
    --learning_rate 3e-8 \
    --evaluate \
    --gpu_id 0 \
    --model_id ./models_local/llama-3.2-1B \
    --lbw_architecture EXTEND \
    --num_bidir_layers 8 \
    --num_unsink_layers 0 \
    --mask_type BACK

*FOR SPECIFIC CONFIGURATION (Qwen3 models):*

Make changes: 
1. --model_id to include the folder where the local model is saved
2. Any further configuration changes for testing

python3 generic_training.py \
    --dataset ace2004 \
    --found_model qwen3_decoder \
    --type_optimization all_encoder_layers_qwen3_decoder \
    --learning_rate 3e-8 \
    --evaluate \
    --gpu_id 0 \
    --model_id ./models_local/Qwen3-Embedding-0.6B \
    --lbw_architecture EXTEND \
    --num_bidir_layers 8 \
    --num_unsink_layers 0 \
    --mask_type BACK

---

## LORA Finetuning

*For finetuning using the BLINK Dataset (Using Dynamic Hard Negative Mining):*

To understand the Dynamic Hard Negative mining:
Official paper - https://aclanthology.org/2025.acl-industry.72/


The setup can be run on multiple gpus instead of just one. 
Choose as many as possible using CUDA_VISIBLE_DEVICES="<gpu_id_0>, <gpu_id_1>, <gpu_id_2>, ..."
Based on the total number of gpus selected, one GPU runs one process. So, if three gpus are selected, replace the number here. --num_processes <total_process_numbers>

Choose the names as per requirements:
LINE 28: DATASET_PATH = "Datasets/blink_training_cleaned_1m.jsonl"
LINE 29: OUTPUT_DIR = "finetuned_models/qwen3-0.6b-lora-dhnm_attn_blink_1m"
LINE 30: LOSS_LOG_PATH = "Loss/qwen3-0.6b_dhnm_attn_mlp_blink_1m.jsonl"

LORA modules selection: LINE 40, LINE 41, LINE 42 (Choose whichever one)

Set further hyperparameters as to liking. LINE 45 - LINE 66


Background run: CUDA_VISIBLE_DEVICES="0,1,2" setsid accelerate launch --num_processes 3 finetuning_dhnm_blink.py > <log_file_name>.log 2>&1 &
To follow progress: tail -f Log/<log_file_name>.log

*For second stage finetuning on AIDA Dataset (Not using the Dynamic Hard Negative Mining):*

The setup can be run on multiple gpus instead of just one. 
Choose as many as possible using CUDA_VISIBLE_DEVICES="<gpu_id_0>, <gpu_id_1>, <gpu_id_2>, ..."
Based on the total number of gpus selected, one GPU runs one process. So, if three gpus are selected, replace the number here. --num_processes <total_process_numbers>

Remove comments and name as per requirements:
LINE 31: DATASET_PATH = "Datasets/aida_finetune.jsonl"
LINE 32: OUTPUT_DIR = "finetuned_models/llama_3.2_1b_blink1m_lora_attn_mlp_aida_stage2_e100"
LINE 33: LOSS_LOG_PATH = "Loss/llama_1b_blink1m_attn_mlp_aida_stage2_no_dhnm_e100.jsonl"

LINE 254 - LINE 292

Comment out: LINE 28 - LINE 30 (Done just for convinience)
             LINE 218 - LINE 250: Lora Config for second stage (Done just for convinience)
             LINE 533 - LINE 539: No Dynamic Hard Negatives Mining (Because it was done in the first stage already, also for time-saving)

LORA modules selection: LINE 40, LINE 41, LINE 42 (Choose whichever one)

Set further hyperparameters as to liking. LINE 45 - LINE 66

Background run: CUDA_VISIBLE_DEVICES="0,1,2" setsid accelerate launch --num_processes 3 finetuning_dhnm_blink.py > <log_file_name>.log 2>&1 &
To follow progress: tail -f Log/<log_file_name>.log



## DATASETS

There are two folders where the datasets are contained.
To go there, paste in terminal
1. cd <git_cloned_folder>/Robust_Ranking/data/aida/wikidata
2. cd <git_cloned_folder>/Robust_Ranking/Finetuning/Datasets

A small example of the blink training dataset is present in the Finetuning/Datasets - blink_training_100.jsonl
It is easier on the Machine RAM.

The Blink dataset structure can be seen from the blink_training_100.jsonl
The AIDA Dataset structure can be viewed from the aida_finetune.jsonl

---

## Training/Evaluation MS MARCO
For E5: python training_msmarco.py --found_model e5 --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0
For biencoder: python training_msmarco.py --found_model biencoder --type_optimization all_encoder_layers --learning_rate 3e-8 --gpu_id 0

There is no support yet for the Decoder based models on the MS MARCO dataset

---

## Noise
the implementation for the noise can be seen in the file optimizers/noise.py
