import json

from os.path import join
from constants import *
from data.base import BertDatasetWrapper, Dataset, Document

def load_oneie_dataset(base_path, tokenizer, remove_doc_with_no_events=True):
    id2split, id2sents = {}, {}

    # Read data files
    for split in ['train', 'dev', 'test']:
        path = join(base_path, '{}.oneie.json'.format(split))
        with open(path, 'r', encoding='utf-8') as r:
            for line in r:
                sent_inst = json.loads(line)
                doc_id = sent_inst['doc_id']
                id2split[doc_id] = split
                # Update id2sents
                if not doc_id in id2sents:
                    id2sents[doc_id] = []
                id2sents[doc_id].append(sent_inst)

    # Parse documents one-by-one
    train, dev, test = Dataset(), Dataset(), Dataset()
    for doc_id in id2sents:
        words_ctx = 0
        sents = id2sents[doc_id]
        sentences, event_mentions, entity_mentions = [], [], []
        for sent_idx, sent in enumerate(sents):
            sentences.append(sent['tokens'])
            # Parse entity mentions
            for entity_mention in sent['entity_mentions']:
                entity_mention['start'] += words_ctx
                entity_mention['end'] += words_ctx
                entity_mentions.append(entity_mention)
            # Parse event mentions
            for event_mention in sent['event_mentions']:
                event_mention['sent_index'] = sent_idx
                event_mention['trigger']['start'] += words_ctx
                event_mention['trigger']['end'] += words_ctx
                event_mentions.append(event_mention)
            # Update words_ctx
            words_ctx += len(sent['tokens'])
        doc = Document(doc_id, sentences, event_mentions, entity_mentions)
        split = id2split[doc_id]
        if split == 'train':
            if not remove_doc_with_no_events or len(event_mentions) > 0:
                train.add_doc(doc)
        if split == 'dev': dev.add_doc(doc)
        if split == 'test': test.add_doc(doc)

    # Verbose
    print('Loaded {} train examples'.format(len(train)))
    print('Loaded {} dev examples'.format(len(dev)))
    print('Loaded {} test examples'.format(len(test)))

    return BertDatasetWrapper(train, tokenizer), \
           BertDatasetWrapper(dev, tokenizer),   \
           BertDatasetWrapper(test, tokenizer)
