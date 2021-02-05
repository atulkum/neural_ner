import os
import time

import torch
import torch.nn as nn

from transformers import Trainer, TrainingArguments
from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from multichoice_dataset import MultiChoiceDataset, variable_collate_fn

class PPAttachmentBert(BertPreTrainedModel):
    def __init__(self, config):
        super(PPAttachmentBert, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, n_heads, labels):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        ones = n_heads.new_ones(n_heads.size(0), torch.max(n_heads))
        range_tensor = ones.cumsum(dim=1)
        label_mask = (n_heads.unsqueeze(1) >= range_tensor).long()

        reshaped_logits = reshaped_logits + (label_mask + 1e-45).log()
        #reshaped_logits = F.log_softmax(reshaped_logits, dim=1)
        _, pred_intent = reshaped_logits.max(dim=1)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)

        return (loss, pred_intent)

def train(data_dir):
    train_dataset = (
        MultiChoiceDataset('train', data_dir)
    )
    eval_dataset = (
        MultiChoiceDataset('dev', data_dir)
    )

    model = PPAttachmentBert.from_pretrained('bert-base-uncased')

    training_args = TrainingArguments(
        output_dir=f'./results_{int(time.time())}',  # output directory
        num_train_epochs=3,  # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        learning_rate=5e-5,
        save_total_limit=1
    )
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=variable_collate_fn,
    )
    trainer.train()
    trainer.save_model()
    result = trainer.evaluate()
    print(result)
def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(p):
    preds = p.predictions
    return {"acc": simple_accuracy(preds, p.label_ids)}

if __name__ == '__main__':
    data_dir = os.path.join(os.path.expanduser('~'),
                              'pp-attachment/dataset/Belinkov2014/pp-data-english')
    train(data_dir)
    ''' 
    dataset = MultiChoiceDataset('train', data_dir)

    dataloader = get_dataloader(dataset, 4, True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(sample_batched)
        break
    '''


