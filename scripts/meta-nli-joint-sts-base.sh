python3 run_meta.py --model_name_or_path bert-base-uncased --model_save_path ./output/meta-nli-joint-sts-base \
  --use_apex_amp --apex_amp_opt_level O1 --force_del --seed 1 --batch_size 64 --max_seq_length 64 --evaluation_steps 200 \
  --concatenation_sent_max_square \
  --add_cl --cl_rate 0.15 --temperature 0.1 --data_augmentation_strategy cutoff --cutoff_direction row --cutoff_rate 0.1 \
  --patience 10