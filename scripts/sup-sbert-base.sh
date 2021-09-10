python3 main.py --model_name_or_path bert-base-uncased --seed 1 --use_apex_amp --apex_amp_opt_level O1 --batch_size 96 \
  --max_seq_length 64 --evaluation_steps 200 --concatenation_sent_max_square \
  --model_save_path ./output/sup-sbert-base --force_del