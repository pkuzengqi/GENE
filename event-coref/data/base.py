import nltk
from utils import convert_to_sliding_window, extract_input_masks_from_mask_windows, flatten

class Document:
    def __init__(self, doc_id, sentences, event_mentions, entity_mentions):
        self.doc_id = doc_id
        self.sentences = sentences
        self.words = flatten(sentences)
        self.event_mentions = event_mentions
        self.entity_mentions = entity_mentions
        self.num_words = len(self.words)

        # Update self.events
        self.events = {}
        for event_mention in event_mentions:
            mention_id = event_mention['id']
            event_id = mention_id[:mention_id.rfind('-')]
            if not event_id in self.events:
                self.events[event_id] = []
            self.events[event_id].append(event_mention)

class Dataset:
    def __init__(self):
        self.data = []

    def add_doc(self, doc):
        self.data.append(doc)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class BertDatasetWrapper:
    def __init__(self, dataset, tokenizer, sliding_window_size = 512):
        self.data = dataset.data
        self.tokenizer = tokenizer

        for doc in self.data:
            doc_tokens = tokenizer.tokenize(' '.join(doc.words))
            doc_token_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
            doc.token_windows, doc.mask_windows = \
                convert_to_sliding_window(doc_token_ids, sliding_window_size, tokenizer)
            doc.input_masks = extract_input_masks_from_mask_windows(doc.mask_windows)

            # Compute the starting index of each word
            doc.word_starts_indexes = []
            for index, word in enumerate(doc_tokens):
                if not word.startswith('##'):
                    doc.word_starts_indexes.append(index)
            assert(len(doc.word_starts_indexes) == len(doc.words))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
