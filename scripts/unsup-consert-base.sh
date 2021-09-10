python3 main.py --no_pair --seed 1 --use_apex_amp --apex_amp_opt_level O1 --batch_size 96 --max_seq_length 64 \
  --evaluation_steps 200 --add_cl --cl_loss_only --cl_rate 0.15 --temperature 0.1 --learning_rate 0.0000005 \
  --train_data stssick --num_epochs 10 --da_final_1 feature_cutoff --da_final_2 shuffle --cutoff_rate_final_1 0.2 \
  --model_name_or_path bert-base-uncased --model_save_path ./output/unsup-consert-base --force_del --no_dropout \
  --patience 10
