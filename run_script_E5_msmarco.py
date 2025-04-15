import os


commands = ["python3 optuna_optimize.py --found_model e5 --dataset msmarco --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 1 --label_smoothness 0.3"]
'''       
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 2 --label_smoothness 0.1",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 2 --label_smoothness 0.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 2 --label_smoothness -0.5",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 2 --label_smoothness -1.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 2 --label_smoothness -2.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 2 --label_smoothness -3.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 2 --label_smoothness -4.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 2 --label_smoothness -5.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 2 --label_smoothness -6.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 2 --label_smoothness -7.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 2 --label_smoothness -8.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 4 --label_smoothness 0.3",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 4 --label_smoothness 0.2",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 4 --label_smoothness 0.1",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 4 --label_smoothness 0.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 4 --label_smoothness -0.5",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 4 --label_smoothness -1.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 4 --label_smoothness -2.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 4 --label_smoothness -3.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 4 --label_smoothness -4.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 4 --label_smoothness -5.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 4 --label_smoothness -6.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 4 --label_smoothness -7.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 4 --label_smoothness -8.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 5 --label_smoothness 0.3",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 5 --label_smoothness 0.2",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 5 --label_smoothness 0.1",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 5 --label_smoothness 0.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 5 --label_smoothness -0.5",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 5 --label_smoothness -1.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 5 --label_smoothness -2.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 5 --label_smoothness -3.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 5 --label_smoothness -4.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 5 --label_smoothness -5.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 5 --label_smoothness -6.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 5 --label_smoothness -7.0",
    "python3 optuna_optimize.py --found_model e5 --dataset lcquad --type_optimization all_encoder_layers_e5 --learning_rate 3e-8 --gpu_id 0 --noise_ratio 5 --label_smoothness -8.0",
]

'''

for cmd in commands:
    print(f"Running: {cmd}")
    os.system(cmd)
