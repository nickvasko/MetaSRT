python3 main.py --continue_training --no_pair --seed 1 --use_apex_amp --apex_amp_opt_level O1 --batch_size 96 \
  --max_seq_length 64 --evaluation_steps 200 --add_cl --cl_loss_only --cl_rate 0.15 --temperature 0.1 \
  --learning_rate 0.0000005 --train_data stssick --num_epochs 25 --data_augmentation_strategy cutoff \
  --cutoff_direction column --cutoff_rate 0.1 --model_name_or_path ./output/sup-sbert-base \
  --model_save_path ./output/sup-consert-sup-unsup-base --force_del --patience 10