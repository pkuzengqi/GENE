import os
import math
import torch
import tqdm
import random

from transformers import *
from models import BasicCorefModel
from scorer import evaluate
from argparse import ArgumentParser
from data import load_oneie_dataset
from utils import RunningAverage, get_configs

def train(config_name, gene_variant=None):
    # Prepare tokenizer, dataset, and model
    configs = get_configs(config_name, verbose=False)
    if configs['use_gene_features']:
        assert(not gene_variant is None)
        configs['gene_variant'] = gene_variant
    tokenizer = BertTokenizer.from_pretrained(configs['transformer'], do_basic_tokenize=False)
    train_set, dev_set, test_set = load_oneie_dataset(configs['base_dataset_path'], tokenizer)
    model = BasicCorefModel(configs)

    # Initialize the optimizer
    num_train_docs = len(train_set)
    epoch_steps = int(math.ceil(num_train_docs / configs['batch_size']))
    num_train_steps = int(epoch_steps * configs['epochs'])
    num_warmup_steps = int(num_train_steps * 0.1)
    optimizer = model.get_optimizer(num_warmup_steps, num_train_steps)
    print('Initialized optimizer')

    # Main training loop
    best_dev_score, iters, batch_loss = 0.0, 0, 0
    for epoch in range(configs['epochs']):
        #print('Epoch: {}'.format(epoch))
        print('\n')
        progress = tqdm.tqdm(total=epoch_steps, ncols=80,
                             desc='Train {}'.format(epoch))
        accumulated_loss = RunningAverage()

        train_indices = list(range(num_train_docs))
        random.shuffle(train_indices)
        for train_idx in train_indices:
            iters += 1
            inst = train_set[train_idx]
            iter_loss = model(inst, is_training=True)[0]
            iter_loss /= configs['batch_size']
            iter_loss.backward()
            batch_loss += iter_loss.data.item()
            if iters % configs['batch_size'] == 0:
                accumulated_loss.update(batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), configs['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = 0
                # Update progress bar
                progress.update(1)
                progress.set_postfix_str('Average Train Loss: {}'.format(accumulated_loss()))
        progress.close()

        # Evaluation after each epoch
        print('Evaluation on the dev set', flush=True)
        dev_score = evaluate(model, dev_set, configs)['avg']

        # Save model if it has better dev score
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            # Evaluation on the test set
            print('Evaluation on the test set', flush=True)
            evaluate(model, test_set, configs)
            # Save the model
            save_path = os.path.join(configs['saved_path'], 'model.pt')
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print('Saved the model', flush=True)

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_name', default='basic')
    parser.add_argument('--gene_variant', default=None)
    args = parser.parse_args()

    # Start training
    train(args.config_name, args.gene_variant)
