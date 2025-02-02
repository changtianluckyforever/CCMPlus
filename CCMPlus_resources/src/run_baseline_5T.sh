#!/usr/bin bash
for seed in 0 2 4
do 
    python -u run_prediction.py \
        --root_path '/scratch/leuven/361/vsc36133/work4luck/processed_datasets' \
        --data_path 'azure_app_T_init_1000.csv' \
        --random_seed $seed \
        --freq '5T' \
        --model 'ccformer' \
        --data 'AZURE' \
        --n_services 1000 \
        --seasonal_patterns 'Hourly' \
        --train_epochs 15 \
        --is_training 1 \
        --patience 5 \
        --learning_rate 0.000001
done
