import time
import random
import numpy as np

import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import random_split

from .dataset_roberta import SpanDataset, variable_collate_fn
from .eval_utils import compute_metrics
from .model import CRFBert

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)

def train():
    all_dataset = SpanDataset('train')
    train_size = int(0.99 * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, eval_dataset = random_split(all_dataset, [train_size, test_size])

    model = CRFBert.from_pretrained('roberta-base')
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameter: {n_params}')

    training_args = TrainingArguments(
        output_dir=f'./results_{int(time.time())}',  # output directory
        num_train_epochs=3,  # total # of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        save_total_limit=1,
        seed=42,
        label_names=["padded_labels"]
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
    #trainer.train()
    #trainer.save_model()
    result = trainer.evaluate()
    for d in data_iter:
        print(d)

if __name__ == '__main__':
    set_seed(42)
    train()



