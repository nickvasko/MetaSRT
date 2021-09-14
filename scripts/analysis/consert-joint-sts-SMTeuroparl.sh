python3 main.py --continue_training --model_name_or_path ./output/sup-consert-joint-sts-base --model_save_path ./output/analysis/consert-joint-sts-SMTeuroparl \
  --use_apex_amp --apex_amp_opt_level O1 --force_del --seed 1 --batch_size 96 --max_seq_length 64 --evaluation_steps 200 \
  --no_pair  --add_cl --cl_loss_only --cl_rate 0.15 --temperature 0.1 --data_augmentation_strategy shuffle \
  --learning_rate 0.0000005 \
  --train_data sts12-SMTeuroparl --num_epochs 25  --patience 5