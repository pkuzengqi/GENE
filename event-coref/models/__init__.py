import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils

from constants import *
from models.helpers import *
from models.base import BaseModel, ScoreModule
from models.encoder import TransformerEncoder

# BasicCorefModel (assuming ground truth event mentions are provided)
class BasicCorefModel(BaseModel):
    def __init__(self, configs):
        BaseModel.__init__(self, configs)

        self.encoder = TransformerEncoder(configs)
        self.pair_scorer = ScoreModule(self.get_pair_embs_size(),
                                      [configs['ffnn_size']] * configs['ffnn_depth'],
                                      configs['dropout_rate'])

        # GENE embeddings (if use_gene_features enabled)
        if configs['use_gene_features']:
            self.gene_dim = GENE2DIM.get(self.configs['gene_variant'], GENE_DIM)
            self.event2emb = get_event2geneemb(configs['gene_variant'])
            for e in self.event2emb:
                self.event2emb[e] = self.event2emb[e].to(self.device)
            self.defaultgene = nn.Embedding(1, self.gene_dim)

        # Initialize embeddings
        for name, param in self.named_parameters():
            if (not 'transformer' in name.lower()) and 'embedding' in name.lower():
                print('Re-initialize embedding {}'.format(name))
                param.data.uniform_(-0.1, 0.1)

        # Move model to device
        self.to(self.device)

    def forward(self, inst, is_training):
        self.train() if is_training else self.eval()

        input_ids = torch.tensor(inst.token_windows).to(self.device)
        input_masks = torch.tensor(inst.input_masks).to(self.device)
        mask_windows = torch.tensor(inst.mask_windows).to(self.device)
        num_windows, window_size = input_ids.size()

        # Apply the Transfomer encoder to get tokens features
        tokens_features = self.encoder(input_ids, input_masks, mask_windows,
                                       num_windows, window_size, is_training).squeeze()
        num_tokens = tokens_features.size()[0]

        # Compute word_features (averaging)
        word_features = []
        word_starts_indexes = inst.word_starts_indexes
        word_ends_indexes = word_starts_indexes[1:] + [num_tokens]
        word_features = get_span_emb(tokens_features, word_starts_indexes, word_ends_indexes)
        assert(word_features.size()[0] == inst.num_words)

        # Compute event_mention_features (averaging)
        event_mentions = inst.event_mentions
        event_mention_starts = [e['trigger']['start'] for e in event_mentions]
        event_mention_ends = [e['trigger']['end'] for e in event_mentions]
        event_mention_features = get_span_emb(word_features, event_mention_starts, event_mention_ends)
        assert(event_mention_features.size()[0] == len(event_mentions))

        # use GENE embeddings (if use_gene_features enabled)
        if self.configs['use_gene_features']:
            gene_embs, index_0 = [], torch.LongTensor([0]).squeeze().to(self.device)
            for e in event_mentions:
                if e['id'] in self.event2emb:
                    gene_embs.append(self.event2emb[e['id']])
                else:
                    gene_embs.append(self.defaultgene(index_0))
            gene_embs = torch.cat([g.unsqueeze(0) for g in gene_embs], dim=0)
            event_mention_features = torch.cat([event_mention_features, gene_embs], dim=1)

        # Compute pair features and score the pairs
        pair_features = self.get_pair_embs(event_mention_features, event_mentions)
        pair_scores = self.pair_scorer(pair_features)

        # Compute antecedent_scores
        k = len(event_mentions)
        span_range = torch.arange(0, k).to(self.device)
        antecedent_offsets = span_range.view(-1, 1) - span_range.view(1, -1)
        antecedents_mask = antecedent_offsets >= 1 # [k, k]
        antecedent_scores = pair_scores + torch.log(antecedents_mask.float())

        # Compute antecedent_labels
        candidate_cluster_ids = self.get_cluster_ids(inst.event_mentions, inst.events)
        same_cluster_indicator = candidate_cluster_ids.unsqueeze(0) == candidate_cluster_ids.unsqueeze(1)
        same_cluster_indicator = same_cluster_indicator & antecedents_mask

        non_dummy_indicator = (candidate_cluster_ids > -1).unsqueeze(1)
        pairwise_labels = same_cluster_indicator & non_dummy_indicator
        dummy_labels = ~pairwise_labels.any(1, keepdim=True)
        antecedent_labels = torch.cat([dummy_labels, pairwise_labels], 1)

        # Compute loss
        dummy_zeros = torch.zeros([k, 1]).to(self.device)
        antecedent_scores = torch.cat([dummy_zeros, antecedent_scores], dim=1)
        gold_scores = antecedent_scores + torch.log(antecedent_labels.float())
        log_norm = logsumexp(antecedent_scores, dim = 1)
        loss = torch.sum(log_norm - logsumexp(gold_scores, dim=1))

        # loss and preds
        top_antecedents = torch.arange(0, k).to(self.device)
        top_antecedents = top_antecedents.unsqueeze(0).repeat(k, 1)
        preds = [torch.tensor(event_mention_starts),
                 torch.tensor(event_mention_ends),
                 top_antecedents,
                 antecedent_scores]

        return loss, preds

    def get_cluster_ids(self, event_mentions, events):
        cluster_ids = []
        non_singleton_clusters = []
        for e in event_mentions:
            mention_id = e['id']
            event_id = mention_id[:mention_id.rfind('-')]
            label = -1
            if len(events[event_id]) > 1:
                if not event_id in non_singleton_clusters:
                    non_singleton_clusters.append(event_id)
                label = non_singleton_clusters.index(event_id)
            cluster_ids.append(label)
        return torch.tensor(cluster_ids).to(self.device)

    def get_pair_embs(self, candidate_embs, event_mentions):
        n, d = candidate_embs.size()
        features_list = []

        # Compute diff_embs and prod_embs
        src_embs = candidate_embs.view(1, n, d).repeat([n, 1, 1])
        target_embs = candidate_embs.view(n, 1, d).repeat([1, n, 1])
        prod_embds = src_embs * target_embs

        # Update features_list
        features_list.append(src_embs)
        features_list.append(target_embs)
        features_list.append(prod_embds)

        # Concatenation
        pair_embs = torch.cat(features_list, 2)

        return pair_embs

    def get_distance_features(self, locations, embeddings, nb_buckets):
        if type(locations) == list:
            locations = torch.tensor(locations).to(self.device)
        offsets = locations.view(-1, 1) - locations.view(1,-1)
        distance_buckets = utils.bucket_distance(offsets, nb_buckets)
        distance_features = embeddings(distance_buckets)
        return distance_features

    def get_span_emb_size(self):
        span_emb_size = self.encoder.transformer_hidden_size
        if self.configs['use_gene_features']:
            self.gene_dim = GENE2DIM.get(self.configs['gene_variant'], GENE_DIM)
            span_emb_size += self.gene_dim
        return span_emb_size

    def get_pair_embs_size(self):
        pair_embs_size = 3 * self.get_span_emb_size() # src_vector, target_vector, product_vector
        return pair_embs_size
