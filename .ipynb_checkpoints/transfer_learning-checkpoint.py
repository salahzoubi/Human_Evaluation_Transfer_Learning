from data_utils import *
from evaluate import *
from dataset import TLDataset
from transformers import T5ForConditionalGeneration,AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
import lightning as pl
import argparse
import logging
import torch
from multiprocessing import cpu_count
from finetuner import T5FineTuner
# from ray_lightning import RayStrategy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import tqdm
from pytorch_lightning import loggers as pl_loggers
import ray
import ast
import os
import gc


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'


# TRG_TASKS = ["OAI", "mauve", "gofigure"]


def main_TL(args):
    # os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu_list}"
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    torch.cuda.empty_cache()
    gc.collect()
    random_seed = args.random_seed
    num_gpus = args.num_gpus
    num_cpus = args.num_cpus
    torch.manual_seed(random_seed)
    # cpu_per_device = num_cpus//num_gpus
    logger.info(f"TASK IDS: {args.task_ids}")

    logger.info(f"Number of available GPU's: {num_gpus}")
    # logger.info(f"Number of available CPU's per device: {cpu_per_device if num_gpus > 0 else num_cpus}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    logger.info(f"Succesfully loaded Tokenizer: {args.model_name_or_path}")
    model = T5FineTuner(args)
    logger.info(f"Succesfully loaded Model: {args.model_name_or_path}")  
    
    #Generate and fine-tune the source tasks based on their id...
    train, val, _ = generate_normal_splits(data_path = "/work/salzubi_umass_edu/human_eval_datasets/master_dataset.csv", random_state=random_seed,  trg_task_ids=args.task_ids, val_split=args.src_val_split, split_size=args.src_test_split, train_fraction=args.src_train_fraction, source_tuning=True, example_threshold=args.example_threshold)
    
    train_data = TLDataset(train, tokenizer)
    logger.info(f"Sucessfully loaded SRC training data: {len(train_data)} examples")
    val_data = TLDataset(val, tokenizer)
    logger.info(f"Sucessfully loaded SRC validation data: {len(val_data)} examples")
    
    train_loader = DataLoader(train_data, batch_size=int(args.train_bsz), drop_last=True , num_workers=num_cpus//num_gpus) # , num_workers=num_cpus//num_gpus, sampler = train_sampler
    val_loader = DataLoader(val_data, batch_size=int(args.eval_bsz), num_workers=num_cpus//num_gpus) #, num_workers=num_cpus//num_gpus

    
    
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"{args.output_dir}logs/{args.file_name}_logs/")
    # strategy = RayStrategy(num_workers=num_gpus, use_gpu=True if num_gpus > 0 else False, find_unused_parameters=False)
    es = EarlyStopping(monitor="val_loss", mode="min", patience=args.src_es_patience)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath = args.output_dir, filename = args.file_name, mode="min")
    val_check_interval = args.val_check_interval
    
    
    if args.use_ckpt != "":
        model = T5FineTuner(hparams = args).load_from_checkpoint(args.use_ckpt, hparams = args)
    else:
        model = T5FineTuner(args)
    trainer = pl.Trainer(max_steps = args.src_num_train_steps, callbacks = [es, checkpoint_callback], val_check_interval=val_check_interval, logger=tb_logger, devices = num_gpus, strategy = "deepspeed_stage_3", precision="bf16", accelerator="gpu") # strategy = strategy
    logger.info("Succesfully loaded model and trainer...")
    # print(f'TRAINING DATA LENGTH: {len(train_data)}')
    # print(f"BATCH SIZE: {args.train_bsz}")
    # print(f'NUMBER OF BATCHES: {len(train_data)//args.train_bsz}')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print(f"BEST MODEL: {checkpoint_callback.best_model_path}")

    trg_model = T5FineTuner(hparams = args).load_from_checkpoint(checkpoint_callback.best_model_path, hparams = args)

    print(f"loaded best model...")
    
    
    
    
    #Generate and fine-tune the target tasks based on their id...
    if args.special_experiment == True:
        train = pd.read_csv(args.train_data_path)
        val = pd.read_csv(args.val_data_path)
        test = pd.read_csv(args.test_data_path)
    else:
        train, val, test = generate_normal_splits(data_path = "/work/salzubi_umass_edu/human_eval_datasets/master_dataset.csv", random_state=random_seed,  trg_task_ids=args.trg_task_ids, train_fraction=args.train_fraction, val_split=args.val_split, split_size=args.test_split, source_tuning=False, example_threshold=args.example_threshold, pseudo_threshold=args.pseudo_threshold)
    
    if args.save_data_pseudo_label == True or args.pseudo_threshold>-1:
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
    logger.info(f"Sucessfully loaded TRG training data: {len(train_data)} examples")
    val_data = TLDataset(val, tokenizer)
    logger.info(f"Sucessfully loaded TRG validation data: {len(val_data)} examples")    
    test_data = TLDataset(test, tokenizer)
    logger.info(f"Sucessfully loaded TRG test data: {len(test_data)} examples")    
    
    train_loader = DataLoader(train_data, batch_size=int(args.train_bsz), drop_last=True, num_workers=num_cpus//num_gpus) # , num_workers=num_cpus//num_gpus, sampler = train_sampler
    val_loader = DataLoader(val_data, batch_size=int(args.eval_bsz), num_workers=num_cpus//num_gpus) #, num_workers=num_cpus//num_gpus
    es = EarlyStopping(monitor="val_loss", mode="min", patience=args.es_patience)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath = args.output_dir, filename = args.file_name, mode="min")
    
    trainer = pl.Trainer(max_steps = args.num_train_steps, strategy=strategy, callbacks = [es, checkpoint_callback], val_check_interval=val_check_interval, logger=tb_logger)
    trainer.fit(trg_model, train_dataloaders=train_loader, val_dataloaders=val_loader) #, accumulate_grad_batches=args.grad_accum
    
    print(f"BEST MODEL: {checkpoint_callback.best_model_path}")

    test_model = T5FineTuner(hparams = args).load_from_checkpoint(checkpoint_callback.best_model_path, hparams = args)
    print(f"loaded best model...")    
    
    
    test_loader = DataLoader(test_data, batch_size=int(args.eval_bsz) , num_workers=12) #, num_workers=num_cpus//num_gpus
    
    out_dir = f"{args.output_dir}{args.file_name}_eval.csv"
    
    pred_df = evaluate_batch(test_loader, test_model, tokenizer, out_dir=out_dir)
    print("finished!!!!")
    
    return pred_df
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_bsz", default=8, type=int)
    parser.add_argument("--eval_bsz", default=16, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--src_train_fraction", default=1,type=float, help="How much of the data, as a fraction, should we use for fine-tuning")

    parser.add_argument("--train_fraction", default=1,type=float, help="How much of the data, as a fraction, should we use for fine-tuning")
    parser.add_argument("--file_name", default="random_model", type=str) #, required=True
    parser.add_argument("--output_dir", default="/work/salzubi_umass_edu/experiments/", type=str) #, required=True
    parser.add_argument("--num_train_steps", default=10000, type=int)
    parser.add_argument("--src_num_train_steps", default=60000, type=int)

    parser.add_argument("--src_es_patience", default=20, type=int)
    parser.add_argument("--es_patience", default=5, type=int)
    parser.add_argument("--val_check_interval", default=.75, type=float)
    parser.add_argument("--dropout", default=.25, type=float)
    parser.add_argument("--random_seed", default=1, type=int)
    parser.add_argument("--num_trials", default=3, type=int)
    parser.add_argument("--task_ids", default = None, type=str, help="list of task_ids that you want to include in the source training...")
    parser.add_argument("--trg_task_ids", default=None, type=str, help="list of trg_task_ids that you want to include in the source training...")
    parser.add_argument("--src_val_split", default=.2, type=float)
    parser.add_argument("--src_test_split", default=.2, type=float)

    parser.add_argument("--val_split", default=.2, type=float)
    parser.add_argument("--test_split", default=.2, type=float)
    parser.add_argument("--num_gpus", default=torch.cuda.device_count(),type=int)
    parser.add_argument("--num_cpus", default=cpu_count(),type=int)
    parser.add_argument("--grad_accum", default=1,type=int)
    parser.add_argument("--gpu_list", default=-1, type=int)
    parser.add_argument("--save_data_pseudo_label",  default=False, action="store_true")
    parser.add_argument("--special_experiment",  default=False, action="store_true")
    parser.add_argument("--train_data_path", default="~/", type=str) #, required=True
    parser.add_argument("--val_data_path", default="~/", type=str) #, required=True
    parser.add_argument("--test_data_path", default="~/", type=str) #, required=True


    parser.add_argument("--pseudo_threshold", default=-1, type=int)

    parser.add_argument("--example_threshold", default=10000, type=int)
    parser.add_argument("--trg_task", choices=["oai", "mauve", "anli", "rankme", "rankgen", "tgoyal_news", "wikisum", "wmt_zh_en", "peer_read", "lens", "dialog_eval", "aggrefact", "par3_fr_en", "stories_wild", "multipit", "mocha", "oai_summ"], default=None, type=str)
    parser.add_argument("--model_name_or_path", choices=["t5-small", "t5-base", "t5-large", "google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"], type=str, default="google/flan-t5-base")
    args = parser.parse_args()
    
    # logging.basicConfig()
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)
    
    main_TL(args)
