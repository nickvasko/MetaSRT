python3 main.py --continue_training --model_name_or_path ./output/sup-consert-joint-sts-base --model_save_path ./output/sup-consert-joint-sts-unsup-base \
  --use_apex_amp --apex_amp_opt_level O1 --force_del --seed 1 --batch_size 96 --max_seq_length 64 --evaluation_steps 200 \
  --concatenation_sent_max_square \
  --train_data stsb --num_epochs 100 --patience 10