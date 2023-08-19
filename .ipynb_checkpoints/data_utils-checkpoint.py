import pandas as pd
import sys
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
import torch
import ray

sys.path.append("~/")

#generates data splits for transfer learning jobs
def generate_transfer_splits(trg_task=None, use_all = False, split_size=.2, train_fraction = 1, data_path = "human_eval_datasets/master_dataset.csv", random_state=1, trg_task_ids=None, task_ids=None, val_split=.15, pseudo_threshold=-1):
    
    df = pd.read_csv(data_path)
    
    if trg_task_ids is not None:
        split = df[df.task_id.isin(trg_task_ids)]
    else:
        split = df[(df.dataset_type == "TRG") & (df.dataset == trg_task)]
    test_split = split.sample(int(split_size * len(split)), random_state=random_state)
    train_val_split = split[~split.index.isin(test_split.index)]
    
    #If you want all training examples (SRC + TRG) to be used for finetuning, then use the use_all flag:
    
#     if use_all == True:
#         train_val_split = pd.concat([df[~df.index.isin(test_split.index)], train_with])
#     else:
#         train_val_split = pd.concat([df[df.dataset_type == "SRC"], train_with])
    
    if task_ids is not None:
        for task_id in task_ids:
            relevant_data = df[df.task_id == task_id]
            train_val_split = pd.concat([relevant_data, train_val_split])
        
    val_split = train_val_split.sample(int(val_split * len(train_val_split) * train_fraction), random_state=random_state)
    train_split = train_val_split[~train_val_split.index.isin(val_split.index)]    
    
    
    if pseudo_threshold > -1:
        train_split = train_split.sample(pseudo_threshold)
    # train_split = train_val_split.sample(int(.9* len(train_val_split) * train_fraction), random_state=random_state)
    # val_split = train_val_split[~train_val_split.index.isin(train_split.index)]
    
    
    return train_split, val_split, test_split

#generates a weighted random sampler...
def generate_weighted_sampler(train_pd):
    
    counter = Counter(train_pd.labels.values)
    label2idx = {x: idx for idx,(x, _) in enumerate(list(counter.items()))}
    class_counts = list(counter.values())
    num_samples = sum(class_counts)
    labels = train_pd.labels.values.tolist()
    class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[label2idx[labels[i]]] for i in range(int(num_samples))]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    
    return sampler

#Generates text given a batch from a dataloader...
def generate_text_batch(batch, model, tokenizer):
    
    input_sample = tokenizer(tokenizer.batch_decode(batch['source_ids'], skip_special_tokens = True), return_tensors = "pt", padding=True, truncation=True).input_ids
    output = model.model.generate(input_sample, max_length = 16, return_dict_in_generate=True, output_scores=True)
    text_output =tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
    
    
    return text_output, output.scores

#Generates data splits for normal finetuning jobs
def generate_normal_splits(trg_task=None, split_size=.2, train_fraction=1, data_path = "human_eval_datasets/master_dataset.csv", random_state=1, val_split=.15, trg_task_ids=None, source_tuning=False, example_threshold=10000, pseudo_threshold=-1):
    
    df = pd.read_csv(data_path)
    
    
    if trg_task_ids is not None:
        split = df[df.task_id.isin(trg_task_ids)]
    else:
        split = df[(df.dataset_type == "TRG") & (df.dataset == trg_task)]
    test_split = split.sample(int(split_size * len(split)), random_state=random_state)
    
    #If you want all training examples (SRC + TRG) to be used for finetuning, then use the use_all flag:
    
    overall_split = split[~split.index.isin(test_split.index)]
    train_val_split = overall_split.sample(int(train_fraction * len(overall_split)))

    val_split = train_val_split.sample(int(val_split * len(train_val_split)), random_state=random_state)
    
    train_split = train_val_split[~train_val_split.index.isin(val_split.index)]
    
    if source_tuning ==True:
        if len(val_split) >= 15000:
            # val_split = val_split.sample(5000)
            val_split = val_split.sample(45000)
        if len(train_split) >= example_threshold:
            train_split = train_split.sample(example_threshold)

        if len(train_split) + len(test_split) >= example_threshold:
            train_split = pd.concat([train_split, test_split]).sample(example_threshold)
        else:
            train_split = pd.concat([train_split, test_split])
            
        return train_split, val_split, None
    
    if pseudo_threshold > -1:
        train_sample = train_split.sample(pseudo_threshold)
        train_split = train_split[~(train_split.index.isin(train_sample.index))]
        return (train_split, train_sample), val_split, test_split
    
    return train_split, val_split, test_split

def process_test_output(batch, test_model, tokenizer, output_scores=False):
    
        y_preds, y_pred_scores = generate_text_batch(batch, test_model, tokenizer)
        y_true = tokenizer.batch_decode(batch['target_ids'], skip_special_tokens=True)
        
        if output_scores is True:
            return y_preds, y_true, y_pred_scores
        return y_preds, y_true
    