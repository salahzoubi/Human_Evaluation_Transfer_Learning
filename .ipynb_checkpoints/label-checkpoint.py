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


output_path_dir = "/home/salzubi_umass_edu/experiments/"
train_bsz = 8
lr = 3e-4
train_fraction = 1
model_name = "google/flan-t5-base"
num_gpus = torch.cuda.device_count()
num_cpus = cpu_count()

args_dict = dict(
    # file_name = file_name,
    # output_dir=output_path_dir, # path to save the checkpoints
    model_name_or_path=model_name,
    tokenizer_name_or_path=model_name,
    max_seq_length=512,
    train_fraction= train_fraction,
    lr=lr,
    weight_decay=0.0,
    scheduler_factor = .1,
    use_gpu = True if num_gpus > 0 else False,
    train_batch_size=train_bsz,
    eval_batch_size=32,
    num_train_steps=10000000,
    es_patience = 4,
    val_check_interval = .25,
    dropout = .2,
    n_gpu=num_gpus,
    cpu_per_device=num_cpus,
    task = "classification",
)

args = argparse.Namespace(**args_dict)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)


from data_utils import *
from evaluate import *

model = load_ckpt(args, f"../experiments/mocha_tests/SRC_task_[50, 33, 40, 32, 31, 51, 39, 52, 53, 57, 64, 54, 55]___TRG_task___None/trial_1/all_src_tasks_TL.ckpt")

test = pd.read_csv("../experiments/mocha_tests/processed_mocha_train_unlabeled.csv")
test_data = TLDataset(test, tokenizer)
test_loader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=32, drop_last=False)

evaluated_batch = evaluate_batch(test_loader, model, tokenizer, out_dir="../experiments/mocha_tests/labeled_train_data_complete.csv")