python3 main.py --model_name_or_path bert-base-uncased --model_save_path ./output/sbert-sts-base \
  --use_apex_amp --apex_amp_opt_level O1 --force_del --seed 1 --batch_size 96 --max_seq_length 64 --evaluation_steps 200 \
  --concatenation_sent_max_square \
  --train_data stsb --num_epochs 100 --patience 10
