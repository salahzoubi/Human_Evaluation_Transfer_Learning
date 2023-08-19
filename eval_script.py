import subprocess
from sklearn.metrics import accuracy_score
import numpy as np
from main import main
import argparse
import ast
import logging
from multiprocessing import cpu_count
import torch
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
    parser.add_argument("--val_check_interval", default=.3, type=float)
    parser.add_argument("--dropout", default=.2, type=float)
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
    parser.add_argument("--baseline",  default=False, action="store_true")
    parser.add_argument("--num_gpus", default=torch.cuda.device_count(),type=int)
    parser.add_argument("--num_cpus", default=cpu_count(),type=int)
    parser.add_argument("--gpu_list", default=-1, type=int)

    parser.add_argument("--trg_task", choices=["oai", "mauve", "anli", "rankme", "rankgen", "tgoyal_news", "wikisum", "wmt_zh_en", "peer_read", "lens", "dialog_eval", "aggrefact", "par3_fr_en", "stories_wild", "multipit", "mocha", "oai_summ"], default=None, type=str)
    parser.add_argument("--model_name_or_path", choices=["t5-small", "t5-base", "t5-large", "google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"], type=str, default="google/flan-t5-base")
    parser.add_argument("--task", default="classification", type=str)
    args = parser.parse_args()
    if args.trg_task_ids is not None:
        args.trg_task_ids = ast.literal_eval(args.trg_task_ids)
        print(f"TRG TASK IDS: {args.trg_task_ids}")
    if args.task_ids is not None:
        print(f"ARGS TASK IDS: {args.task_ids}")
        args.task_ids = ast.literal_eval(args.task_ids)
        if args.trg_task is not None:
            file_name =f"SRC_task_{args.task_ids}___TRG_task___{args.trg_task}"
        else:
            file_name =f"SRC_task_{args.task_ids}___TRG_task___{args.trg_task_ids}"

    else:
        if args.trg_task is not None:
            file_name =f"TRG_task___{args.trg_task}_baseline_dropout_{args.dropout}"
        else:
            file_name=f"TRG_task___{args.trg_task_ids}_baseline_dropout_{args.dropout}"
        
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)    
    
    file_output_dir = args.output_dir
    NUM_TRIALS=args.num_trials
    
    avg_acc = []
    for i in range(1, NUM_TRIALS+1):
        
        logger.info(f"STARTING TRIAL: {i}")
        args.random_seed = i
        args.output_dir = f"{file_output_dir}{file_name}/trial_{i}/"
        args.file_name = f"{file_name}___trial_{args.random_seed}"
        # if args.special_experiment == True:
        #     args.test_data_path = f"../experiments/mocha_tests/processed_sample_{i}.csv"
        pred_df = main(args)
        accuracy = accuracy_score(pred_df.predicted, pred_df.ground_truth)
        avg_acc += [accuracy]
        logger.info(f"TRIAL: {i} COMPLETED\nACCURACY: {accuracy}")

    with open(f"{file_output_dir}{file_name}/FINAL_EVAL.txt", "w") as file:
        file.write(f"The {NUM_TRIALS} different accuracies are: {avg_acc}\n")
        file.write(f"\nAVG ACC: {np.mean(avg_acc)}\n")
        file.write(f"STD: {np.std(avg_acc)}")

        
    