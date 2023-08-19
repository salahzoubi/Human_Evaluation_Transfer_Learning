import torch
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration,AutoTokenizer, AutoConfig, RobertaForSequenceClassification, get_constant_schedule_with_warmup, AdamW
from transformers.optimization import Adafactor, AdafactorSchedule
import torch_optimizer as optim

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.h_params = hparams
        config = AutoConfig.from_pretrained(hparams.model_name_or_path)
        config.dropout = hparams.dropout
        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path, config = self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.model_name_or_path)
        self.task = hparams.task

  
    def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
          input_ids=batch["source_ids"],
          attention_mask=batch["source_mask"],
          labels=labels,
          decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        
    def predict_step(self, batch, batch_idx, max_length = 8):
        input_sample = self.tokenizer(self.tokenizer.batch_decode(batch['source_ids'], skip_special_tokens = True), return_tensors = "pt", padding=True, truncation=True).input_ids
        output = self.model.generate(input_sample, max_length = max_length, return_dict_in_generate=True, output_scores=True)
        text_output = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        
        {"outputs": text_output ,"scores": output.scores, "ground_truth": batch["target_ids"]}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.h_params.lr)
        # optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True)
        
        # scheduler = WarmupLR(optimizer)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1)
        # scheduler = get_constant_schedule_with_warmup(optimizer, 100)
        # scheduler = AdafactorSchedule(optimizer)

        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}