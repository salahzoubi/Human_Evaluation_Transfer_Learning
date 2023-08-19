from sklearn.metrics import accuracy_score
from finetuner import T5FineTuner
import pytorch_lightning as pl
from data_utils import process_test_output
from tqdm import tqdm
import pandas as pd
#given a batch of test examples, it will generate the answers for each....

def generate_text_from_example(example, model, tokenizer, max_length = 16):
    
    input_sample = tokenizer(tokenizer.batch_decode(example['source_ids'], skip_special_tokens = True), return_tensors = "pt", padding=True, truncation=True).input_ids
    output = model.model.generate(input_sample, max_length = max_length)
    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    
    return generated_text


def load_ckpt(args, ckpt_path):
    
    model = T5FineTuner(hparams = args).load_from_checkpoint(ckpt_path, hparams = args)
    return model

def evaluate_batch(test_loader, test_model, tokenizer, out_dir=None, output_scores=False):
    
    y_preds = []
    y_true= []
    if output_scores is True:
        y_scores = []
        
    for idx, batch in tqdm(enumerate(test_loader)):
        if output_scores is True:
            y_pred, y, scores = process_test_output(batch, test_model, tokenizer, output_scores)
        else:
            y_pred, y = process_test_output(batch, test_model, tokenizer, output_scores)
        
        y_preds += y_pred
        y_true += y
        if output_scores is True:
            y_scores += scores
    
    final_df = pd.DataFrame({"predicted": y_preds, "ground_truth": y_true})
    
    if out_dir is not None:
        final_df.to_csv(out_dir, index=False)
    
    if output_scores is True:
        return final_df, y_scores
    return final_df
    
    