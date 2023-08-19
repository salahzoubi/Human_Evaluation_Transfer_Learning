from data_utils import *
from evaluate import *
from dataset import TLDataset
from transformers import T5ForConditionalGeneration,AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import argparse
import logging
import torch
from multiprocessing import cpu_count
from finetuner import T5FineTuner
from ray_lightning import RayStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from pytorch_lightning import loggers as pl_loggers
import ray
import ast
import os
import gc
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'


# TRG_TASKS = ["OAI", "mauve", "gofigure"]


def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu_list}"
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    torch.cuda.empty_cache()
    gc.collect()
    random_seed = args.random_seed
    num_gpus = args.num_gpus
    num_cpus = args.num_cpus
    # cpu_per_device = num_cpus//num_gpus
    torch.manual_seed(random_seed)
    logger.info(f"TASK IDS: {args.task_ids}")
    logger.info(f"TRANSFER LEARNING SETTING: {args.TL}")

    logger.info(f"Number of available GPU's: {num_gpus}")
    # logger.info(f"Number of available CPU's per device: {cpu_per_device if num_gpus > 0 else num_cpus}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    logger.info(f"Succesfully loaded Tokenizer: {args.model_name_or_path}")
    model = T5FineTuner(args)
    logger.info(f"Succesfully loaded Model: {args.model_name_or_path}")  
    
    if args.TL==True:
        train, val, test = generate_transfer_splits(args.trg_task, data_path = "/work/salzubi_umass_edu/human_eval_datasets/master_dataset.csv", random_state=random_seed, train_fraction = args.train_fraction, task_ids=args.task_ids, trg_task_ids=args.trg_task_ids, val_split=args.val_split, split_size=args.test_split)
    
    elif args.special_experiment==True:
        train = pd.read_csv(args.train_data_path)
        val = pd.read_csv(args.val_data_path)
        test = pd.read_csv(args.test_data_path)
    else:
        train, val, test = generate_normal_splits(args.trg_task, data_path = "/work/salzubi_umass_edu/human_eval_datasets/master_dataset.csv", random_state=random_seed,  trg_task_ids=args.trg_task_ids, val_split=args.val_split, split_size=args.test_split,    pseudo_threshold=args.pseudo_threshold)
    
    
    if args.save_data_pseudo_label == True:
        Path(f"{args.output_dir}{args.file_name}").mkdir(parents=True, exist_ok=True)
        train[1].to_csv(f"{args.output_dir}{args.file_name}_state_{random_seed}_train_labeled.csv" ,index=True)
        train[0].to_csv(f"{args.output_dir}{args.file_name}_state_{random_seed}_train_unlabeled.csv" ,index=True)
        val.to_csv(f"{args.output_dir}{args.file_name}_state_{random_seed}_val.csv" ,index=True)
        test.to_csv(f"{args.output_dir}{args.file_name}_state_{random_seed}_test.csv" ,index=True)
    if args.pseudo_threshold>-1:
        train = train[1]
    else:
        if len(train) == 2:
            train=train[0]
    
    train_data = TLDataset(train, tokenizer)
    logger.info(f"Sucessfully loaded training data: {len(train_data)} examples")
    val_data = TLDataset(val, tokenizer)
    logger.info(f"Sucessfully loaded validation data: {len(val_data)} examples")
    test_data = TLDataset(test, tokenizer)
    logger.info(f"Sucessfully loaded test data: {len(test_data)} examples")
    # train_sampler = generate_weighted_sampler(train)
        
    train_loader = DataLoader(train_data, batch_size=int(args.train_bsz), drop_last=True) # , num_workers=num_cpus//num_gpus, sampler = train_sampler
    val_loader = DataLoader(val_data, batch_size=int(args.eval_bsz)) #, num_workers=num_cpus//num_gpus

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"{args.output_dir}logs/{args.file_name}_logs/")
    strategy = RayStrategy(num_workers=num_gpus, use_gpu=True if num_gpus > 0 else False, find_unused_parameters=False)
    es = EarlyStopping(monitor="val_loss", mode="min", patience=args.es_patience)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath = args.output_dir, filename = args.file_name, mode="min")
    val_check_interval = args.val_check_interval
    
    
    model = T5FineTuner(args)
    trainer = pl.Trainer(max_steps = args.num_train_steps, strategy=strategy, callbacks = [es, checkpoint_callback], val_check_interval=val_check_interval, logger=tb_logger) # strategy = strategy
    logger.info("Succesfully loaded model and trainer...")
        
    
    trainer.fit(model, train_loader, val_loader)
    
    print(f"BEST MODEL: {checkpoint_callback.best_model_path}")

    test_model = T5FineTuner(hparams = args).load_from_checkpoint(checkpoint_callback.best_model_path, hparams = args)
    print(f"loaded best model...")
    test_loader = DataLoader(test_data, batch_size=int(args.eval_bsz), num_workers = num_cpus) #, num_workers=num_cpus//num_gpus
    
    out_dir = f"{args.output_dir}{args.file_name}_eval.csv"
    pred_df = evaluate_batch(test_loader, test_model, tokenizer, out_dir=out_dir)
    print("finished!!!!")
    
    return pred_df
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_bsz", default=8, type=int)
    parser.add_argument("--eval_bsz", default=16, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--train_fraction", default=1,type=float, help="How much of the data, as a fraction, should we use for fine-tuning")
    parser.add_argument("--file_name", default="random_model", type=str) #, required=True
    parser.add_argument("--output_dir", default="/work/salzubi_umass_edu/experiments/", type=str) #, required=True
    parser.add_argument("--num_train_steps", default=3e6, type=int)
    parser.add_argument("--es_patience", default=12, type=int)
    parser.add_argument("--val_check_interval", default=.75, type=float)
    parser.add_argument("--dropout", default=.25, type=float)
    parser.add_argument("--random_seed", default=1, type=int)
    parser.add_argument("--num_trials", default=3, type=int)
    parser.add_argument("--TL", default=False, action="store_true", help="Whether or not the task is a transfer learning task. Has an effect on which dataset is loaded...")
    parser.add_argument("--task_ids", default = None, type=str, help="list of task_ids that you want to include in the source training...")
    parser.add_argument("--trg_task_ids", default=None, type=str, help="list of trg_task_ids that you want to include in the source training...")
    parser.add_argument("--val_split", default=.15, type=float)
    parser.add_argument("--test_split", default=.2, type=float)
    parser.add_argument("--save_data_pseudo_label",  default=False, action="store_true")
    parser.add_argument("--special_experiment",  default=False, action="store_true")
    parser.add_argument("--train_data_path", default="~/", type=str) #, required=True
    parser.add_argument("--val_data_path", default="~/", type=str) #, required=True
    parser.add_argument("--test_data_path", default="~/", type=str) #, required=True
    parser.add_argument("--pseudo_threshold", default=-1, type=int)
    parser.add_argument("--num_gpus", default=torch.cuda.device_count(),type=int)
    parser.add_argument("--num_cpus", default=cpu_count(),type=int)
    parser.add_argument("--gpu_list", default=-1, type=int)

    parser.add_argument("--trg_task", choices=["oai", "mauve", "anli", "rankme", "rankgen", "tgoyal_news", "wikisum", "wmt_zh_en", "peer_read", "lens", "dialog_eval", "aggrefact", "par3_fr_en", "stories_wild", "multipit", "mocha", "oai_summ"], default=None, type=str)
    parser.add_argument("--model_name_or_path", choices=["t5-small", "t5-base", "t5-large", "google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"], type=str, default="google/flan-t5-base")
    parser.add_argument("--task", default="classification", type=str)
    args = parser.parse_args()
    
    # logging.basicConfig()
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)
    
    main(args)
