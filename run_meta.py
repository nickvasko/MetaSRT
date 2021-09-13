"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_nli.py --seed 1234

OR
python training_nli.py --seed 1234 --model_name_or_path bert-base-uncased
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction, SentenceEvaluator
from sentence_transformers.util import batch_to_device
import logging
import sys
import os
import json
import copy
import gzip
import csv
import random
import torch
import numpy as np
import argparse
import shutil
from copy import deepcopy

from tensorboardX import SummaryWriter
from eval import eval_nli_unsup
from data_utils import load_datasets, save_samples, load_senteval_binary, load_senteval_sst, load_senteval_trec, \
    load_senteval_mrpc, load_chinese_tsv_data
from correlation_visualization import corr_visualization
from transformers import AdamW
from tqdm.autonotebook import tqdm, trange
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional

logging.basicConfig(format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def parse_args():
    """
    Argument settings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str,
                        choices=["sst2", "trec", "mrpc", "mr", "cr", "subj", "mpqa", "nli", "stssick", "stsb"],
                        default="nli", help="Training data, on NLI or STS dataset")
    parser.add_argument("--no_pair", action="store_true", help="If provided, do not pair two training texts")
    parser.add_argument("--data_proportion", type=float, default=1.0, help="The proportion of training dataset")
    parser.add_argument("--do_upsampling", action="store_true",
                        help="If provided, do upsampling to original size of training dataset")
    parser.add_argument("--no_shuffle", action="store_true", help="If provided, do not shuffle the training data")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducing experimental results")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased",
                        help="The model path or model name of pre-trained model")
    parser.add_argument("--continue_training", action="store_true",
                        help="Whether to continue training or just from BERT")
    parser.add_argument("--model_save_path", type=str, default=None, help="Custom output dir")
    parser.add_argument("--tensorboard_log_dir", type=str, default=None, help="Custom tensorboard log dir")
    parser.add_argument("--force_del", action="store_true",
                        help="Delete the existing save_path and do not report an error")

    parser.add_argument("--use_apex_amp", action="store_true", help="Use apex amp or not")
    parser.add_argument("--apex_amp_opt_level", type=str, default=None, help="The opt_level argument in apex amp")

    parser.add_argument("--batch_size", type=int, default=16, help="Training mini-batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="The learning rate")
    parser.add_argument("--evaluation_steps", type=int, default=1000, help="The steps between every evaluations")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The max sequence length")
    parser.add_argument("--loss_rate_scheduler", type=int, default=0,
                        help="The loss rate scheduler, default strategy 0 (i.e. do nothing, see AdvCLSoftmaxLoss for more details)")
    parser.add_argument("--no_dropout", action="store_true", help="Add no dropout when training")

    parser.add_argument("--concatenation_sent_max_square", action="store_true",
                        help="Concat max-square features of two text representations when training classification")

    parser.add_argument("--add_cl", action="store_true", help="Use contrastive loss or not")
    parser.add_argument("--data_augmentation_strategy", type=str, default="adv",
                        choices=["adv", "none", "meanmax", "shuffle", "cutoff", "shuffle-cutoff", "shuffle+cutoff",
                                 "shuffle_embeddings"], help="The data augmentation strategy in contrastive learning")
    parser.add_argument("--cutoff_direction", type=str, default=None,
                        help="The direction of cutoff strategy, row, column or random")
    parser.add_argument("--cutoff_rate", type=float, default=None, help="The rate of cutoff strategy, in (0.0, 1.0)")
    parser.add_argument("--cl_loss_only", action="store_true",
                        help="Ignore the main task loss (e.g. the CrossEntropy loss) and use the contrastive loss only")
    parser.add_argument("--cl_rate", type=float, default=0.01, help="The contrastive loss rate")
    parser.add_argument("--regularization_term_rate", type=float, default=0.0,
                        help="The loss rate of regularization term for contrastive learning")
    parser.add_argument("--cl_type", type=str, default="nt_xent", help="The contrastive loss type, nt_xent or cosine")
    parser.add_argument("--temperature", type=float, default=0.5, help="The temperature for contrastive loss")
    parser.add_argument("--mapping_to_small_space", type=int, default=None,
                        help="Whether to mapping sentence representations to a low dimension space (similar to SimCLR) and give the dimension")
    parser.add_argument("--add_contrastive_predictor", type=str, default=None,
                        help="Whether to use a predictor on one side (similar to SimSiam) and give the projection added to which side (normal or adv)")
    parser.add_argument("--add_projection", action="store_true",
                        help="Add projection layer before predictor, only be considered when add_contrastive_predictor is not None")
    parser.add_argument("--projection_norm_type", type=str, default=None,
                        help="The norm type used in the projection layer beforn predictor")
    parser.add_argument("--projection_hidden_dim", type=int, default=None,
                        help="The hidden dimension of the projection or predictor MLP")
    parser.add_argument("--projection_use_batch_norm", action="store_true",
                        help="Whether to use batch normalization in the hidden layer of MLP")
    parser.add_argument("--contrastive_loss_stop_grad", type=str, default=None,
                        help="Use stop gradient to contrastive loss (and which mode to apply) or not")

    parser.add_argument("--da_final_1", type=str, default=None,
                        help="The final 5 data augmentation strategies for view1 (none, shuffle, token_cutoff, feature_cutoff, dropout, span)")
    parser.add_argument("--da_final_2", type=str, default=None,
                        help="The final 5 data augmentation strategies for view2 (none, shuffle, token_cutoff, feature_cutoff, dropout, span)")
    parser.add_argument("--cutoff_rate_final_1", type=float, default=None,
                        help="The final cutoff/dropout rate for view1")
    parser.add_argument("--cutoff_rate_final_2", type=float, default=None,
                        help="The final cutoff/dropout rate for view2")

    parser.add_argument("--patience", default=None, type=int, help="The patience for early stop")

    return parser.parse_args()


def set_seed(seed: int, for_multi_gpu: bool):
    """
    Added script to set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if for_multi_gpu:
        torch.cuda.manual_seed_all(seed)


def fit(model, meta_model, task_dataloader, meta_dataloader, task_loss_model, meta_loss_model,
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        scheduler: str = 'WarmupLinear',
        warmup_steps: int = 10000,
        optimizer_params: Dict[str, object] = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_apex_amp: bool = False,
        apex_amp_opt_level: str = None,
        callback: Callable[[float, int, int], None] = None,
        early_stop_patience: Optional[int] = None,
        meta_learning_rate: float = 5e-7
        ):
    """
    Train the model with the given training objective
    Each training objective is sampled in turn for one batch.
    We sample only as many batches from each objective as there are in the smallest one
    to make sure of equal training with each dataset.

    :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
    :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
    :param epochs: Number of epochs for training
    :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
    :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
    :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
    :param optimizer_class: Optimizer
    :param optimizer_params: Optimizer parameters
    :param weight_decay: Weight decay for model parameters
    :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
    :param output_path: Storage path for the model and evaluation files
    :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
    :param max_grad_norm: Used for gradient normalization.
    :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
    :param callback: Callback function that is invoked after each evaluation.
            It must accept the following three parameters in this order:
            `score`, `epoch`, `steps`
    :param output_path_ignore_not_empty: deprecated, no longer used
    """

    if use_apex_amp:
        from apex import amp

    model.to(model._target_device)
    meta_model.to(meta_model._target_device)
    task_loss_model.to(model._target_device)
    meta_loss_model.to(meta_model._target_device)

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    # Use smart batching
    task_dataloader.collate_fn = model.smart_batching_collate
    meta_dataloader.collate_fn = model.smart_batching_collate

    model.best_score = -9999999
    meta_model.best_score = -9999999

    steps_per_epoch = len(task_dataloader)

    num_train_steps = int(steps_per_epoch * epochs)

    # Prepare optimizers
    param_optimizer = list(task_loss_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_params)
    scheduler = model._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

    model.global_step = 0
    meta_model.global_step = 0

    task_iterator = iter(task_dataloader)
    meta_iterator = iter(meta_dataloader)

    if use_apex_amp:
        task_loss_model, optimizer = amp.initialize(task_loss_model, optimizer, opt_level=apex_amp_opt_level)

    skip_scheduler = False
    best_dev_score = -9999999
    patience = early_stop_patience
    for epoch in trange(epochs, desc="Epoch"):
        training_steps = 0

        task_loss_model.zero_grad()
        task_loss_model.train()

        for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):
            try:
                task_data = next(task_iterator)
            except StopIteration:
                task_iterator = iter(task_dataloader)
                task_data = next(task_iterator)
            try:
                meta_data = next(meta_iterator)
            except StopIteration:
                meta_iterator = iter(meta_dataloader)
                meta_data = next(meta_iterator)

            features, labels = batch_to_device(task_data, model._target_device)
            loss_value = task_loss_model(features, labels)
            model.tensorboard_writer.add_scalar(f"train_loss_{0}", loss_value.item(), global_step=model.global_step)
            if use_apex_amp:
                with amp.scale_loss(loss_value, optimizer) as scaled_loss_value:
                    scaled_loss_value.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
            else:
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(task_loss_model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if not skip_scheduler:
                scheduler.step()

            meta_loss_model.model[0].load_state_dict(task_loss_model.model[0].state_dict())
            features, labels = batch_to_device(meta_data, model._target_device)
            loss_value = meta_loss_model(features, labels)
            model.tensorboard_writer.add_scalar(f"meta_loss_{0}", loss_value.item(), global_step=model.global_step)
            loss_value.backward()

            for model_p, meta_p in zip(model.parameters(), meta_loss_model.parameters()):
                if meta_p.grad is not None:
                    model_p.data = model_p.data - meta_learning_rate * meta_p.grad

            meta_loss_model.zero_grad()
            task_loss_model.model[0].load_state_dict(model[0].state_dict())

            training_steps += 1
            model.global_step += 1
            meta_model.global_step += 1
            if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                score = model._eval_during_training(evaluator, output_path, save_best_model, epoch,
                                                   training_steps, callback)
                if score is not None and early_stop_patience is not None:
                    if score > best_dev_score:
                        best_dev_score = score
                        patience = early_stop_patience
                    else:
                        patience -= 1
                        logging.info(
                            f"No improvement over previous best score ({score:.6f} vs {best_dev_score:.6f}), patience = {patience}")
                        if patience == 0:
                            logging.info("Run out of patience, early stop")
                            return
                task_loss_model.zero_grad()
                task_loss_model.train()

        score = model._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)
        if score is not None and early_stop_patience is not None:
            if score > best_dev_score:
                best_dev_score = score
                patience = early_stop_patience
            else:
                patience -= 1
                logging.info(
                    f"No improvement over previous best score ({score:.6f} vs {best_dev_score:.6f}), patience = {patience}")
                if patience == 0:
                    logging.info("Run out of patience, early stop")
                    return


def main(args):
    logging.info(f"Training arguments: {args.__dict__}")

    set_seed(args.seed, for_multi_gpu=False)

    # Check if dataset exists. If not, download and extract  it
    nli_dataset_path = 'datasets/AllNLI.tsv.gz'
    sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
    if not os.path.exists(nli_dataset_path):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)
    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

    # Read the dataset
    train_batch_size = args.batch_size

    bert_model_type_str = "base" if "base" in args.model_name_or_path else "large"

    cl_mapping_to_lower_str = "" if args.mapping_to_small_space is None else f"-simclr-{args.projection_hidden_dim}-{args.mapping_to_small_space}-{'bn' if args.projection_use_batch_norm else ''}"
    cl_add_predictor_str = "" if args.add_contrastive_predictor is None else f"-simsiam{'p' if args.add_projection else ''}{args.projection_norm_type if args.projection_norm_type is not None else ''}-{args.projection_hidden_dim}-{args.add_contrastive_predictor}-{'bn' if args.projection_use_batch_norm else ''}"
    cl_type_str = "" if args.cl_type == "nt_xent" else "-cosine"
    cl_param_str = "" if not args.add_cl else f"cl-rate{args.cl_rate}-t{args.temperature}{'-stopgrad' + args.contrastive_loss_stop_grad if args.contrastive_loss_stop_grad else ''}{cl_mapping_to_lower_str}{cl_add_predictor_str}{cl_type_str}_"

    model_save_path = args.model_save_path or os.path.join("./output",
                                                           f"{args.train_data}_bert-{bert_model_type_str}_{args.batch_size}-{args.num_epochs}_{'maxsqr_' if args.concatenation_sent_max_square else ''}{'stopgrad_' if args.normal_loss_stop_grad else ''}{adv_param_str}{cl_param_str}seed={args.seed}")

    if os.path.exists(model_save_path):
        if args.force_del:
            shutil.rmtree(model_save_path)
            os.mkdir(model_save_path)
        else:
            raise ValueError("Existing output_dir for save model")
    else:
        os.mkdir(model_save_path)

    # Tensorboard writer
    tensorboard_writer = SummaryWriter(args.tensorboard_log_dir or os.path.join(model_save_path, "logs"))

    with open(os.path.join(model_save_path, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4, ensure_ascii=False)
    with open(os.path.join(model_save_path, "command.txt"), "w") as f:
        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        f.write(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python3 {' '.join(sys.argv)}")

    if args.continue_training:
        if args.no_dropout:
            sentence_bert_config_path = os.path.join(args.model_name_or_path, "0_Transformer",
                                                     "sentence_bert_config.json")
            sentence_bert_config_dict = json.load(open(sentence_bert_config_path, "r"))
            # change config
            new_config = copy.deepcopy(sentence_bert_config_dict)
            new_config["attention_probs_dropout_prob"] = 0.0
            new_config["hidden_dropout_prob"] = 0.0
            json.dump(new_config, open(sentence_bert_config_path, "w"), indent=2)
            # load model
            model = SentenceTransformer(args.model_name_or_path)
            meta_model = SentenceTransformer(args.model_name_or_path)
            # recover config
            json.dump(sentence_bert_config_dict, open(sentence_bert_config_path, "w"), indent=2)
        else:
            model = SentenceTransformer(args.model_name_or_path)
            meta_model = SentenceTransformer(args.model_name_or_path)
    else:
        # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
        if args.no_dropout:
            word_embedding_model = models.Transformer(args.model_name_or_path, attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0)
        else:
            word_embedding_model = models.Transformer(args.model_name_or_path)
        # Apply mean pooling to get one fixed sized sentence vector
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        meta_model = SentenceTransformer(modules=[deepcopy(word_embedding_model), deepcopy(pooling_model)])
    model.tensorboard_writer = tensorboard_writer
    model.max_seq_length = args.max_seq_length
    meta_model.tensorboard_writer = tensorboard_writer
    meta_model.max_seq_length = args.max_seq_length

    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    # Read the AllNLI.tsv.gz file and create the training dataset
    logging.info("Read AllNLI train dataset")
    train_samples = []
    with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'train':
                label_id = label2int[row['label']]
                if args.no_pair:
                    assert args.cl_loss_only, "no pair texts only used when contrastive loss only"
                    train_samples.append(InputExample(texts=[row['sentence1']]))
                    train_samples.append(InputExample(texts=[row['sentence2']]))
                else:
                    train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
    save_samples(train_samples, os.path.join(model_save_path, "train_task_texts.txt"))
    train_dataset = SentencesDataset(train_samples, model=model)
    task_dataloader = DataLoader(train_dataset, shuffle=not args.no_shuffle, batch_size=train_batch_size)

    # Read STSb training dataset
    logging.info("Read STS Benchmark train dataset")
    train_samples = load_datasets(datasets=["stsb"], need_label=True, use_all_unsupervised_texts=False, no_pair=args.no_pair)
    save_samples(train_samples, os.path.join(model_save_path, "train_meta_texts.txt"))
    train_dataset = SentencesDataset(train_samples, model=model)
    meta_dataloader = DataLoader(train_dataset, shuffle=not args.no_shuffle, batch_size=train_batch_size)

    task_loss = losses.AdvCLSoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int), concatenation_sent_max_square=args.concatenation_sent_max_square, use_contrastive_loss=args.add_cl, contrastive_loss_type=args.cl_type, contrastive_loss_rate=args.cl_rate, temperature=args.temperature, contrastive_loss_stop_grad=args.contrastive_loss_stop_grad, mapping_to_small_space=args.mapping_to_small_space, add_contrastive_predictor=args.add_contrastive_predictor, projection_hidden_dim=args.projection_hidden_dim, projection_use_batch_norm=args.projection_use_batch_norm, add_projection=args.add_projection, projection_norm_type=args.projection_norm_type, contrastive_loss_only=args.cl_loss_only, data_augmentation_strategy=args.data_augmentation_strategy, cutoff_direction=args.cutoff_direction, cutoff_rate=args.cutoff_rate, no_pair=args.no_pair, regularization_term_rate=args.regularization_term_rate, loss_rate_scheduler=args.loss_rate_scheduler, data_augmentation_strategy_final_1=args.da_final_1, data_augmentation_strategy_final_2=args.da_final_2, cutoff_rate_final_1=args.cutoff_rate_final_1, cutoff_rate_final_2=args.cutoff_rate_final_2)
    meta_loss = losses.CosineSimilarityLoss(model=model)

    # Read STSbenchmark dataset and use it as development set
    logging.info("Read STSbenchmark dev dataset")
    dev_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'dev':
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                     name='sts-dev',
                                                                     main_similarity=SimilarityFunction.COSINE)

    # Configure the training
    num_epochs = args.num_epochs

    model.num_steps_total = math.ceil(len(train_dataset) * num_epochs / train_batch_size)
    warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    fit(model, meta_model, task_dataloader, meta_dataloader, task_loss, meta_loss,
        evaluator=dev_evaluator,
        epochs=num_epochs,
        optimizer_params={'lr': args.learning_rate, 'eps': 1e-6, 'correct_bias': False},
        evaluation_steps=args.evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        use_apex_amp=args.use_apex_amp,
        apex_amp_opt_level=args.apex_amp_opt_level,
        early_stop_patience=args.patience)

    # Test on STS Benchmark
    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'test':
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
    if args.chinese_dataset != "none":
        test_samples = load_chinese_tsv_data(args.chinese_dataset, "test")
    model = SentenceTransformer(model_save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size,
                                                                      name='sts-test',
                                                                      main_similarity=SimilarityFunction.COSINE)
    test_evaluator(model, output_path=model_save_path)

    # Test on unsupervised dataset (mainly STS related dataset)
    eval_nli_unsup(model_save_path, main_similarity=SimilarityFunction.COSINE)
    eval_nli_unsup(model_save_path, main_similarity=SimilarityFunction.COSINE, last2avg=True)
    corr_visualization(model_save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
