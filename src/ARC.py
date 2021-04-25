import numpy as np
import torch
import torch.nn as nn
from utils import vocab_size_dict


class ARC_trainer(nn.Module):
    def __init__(self, G, valG, nemb=256, batch_size=16):
        super().__init__()
        self.batch_size = batch_size

        # W_r for each relation

        self.classifiers = nn.ModuleDict({
            etype: ARC(nemb) for etype in G.etypes
        })



        self.samples = dict()
        self.val_samples = dict()
        self.sample(G,valG)

    def sample(self,G,valG):
        for etype in G.etypes:
            self.samples[etype] = self.presample(G, etype)
        for etype in valG.etypes:
            self.val_samples[etype] = self.presample(valG, etype)



    def forward(self, G, emb, batch_num=0):
        # input emb = {'event': [3721,768], 'entity':[3688,768]}
        arc_loss = []
        if emb['event'].shape[0] == vocab_size_dict['train']['event']:
            samples = self.samples
        else:
            samples = self.val_samples


        for etype in G.etypes:

            ### get pre-sampled results ###
            try:
                (x_id, pos_id, neg_id) = samples[etype]
            except:
                # one bug: some etype may not be initialized; this model should be initialized with type file
                continue


            ### get batches ###
            st = batch_num*self.batch_size % x_id.shape[0]
            en = (batch_num+1)*self.batch_size % x_id.shape[0]
            if en == 0 and st == 0:
                en = -1
            x_id = x_id[st:en]
            pos_id = pos_id[st:en]
            neg_id = neg_id[st:en]

            if x_id.shape[0] < 1:
                continue
            # if x_id.shape[0] > self.batch_size:
            #     batch_id = self.batchsample(pos_id, self.batch_size)
            #     x_id, pos_id = x_id[batch_id], pos_id[batch_id]

            ### get embedding ###
            if etype.startswith('Event2Entity'):
                x, pos, neg = emb['event'][x_id], emb['entity'][pos_id], emb['entity'][neg_id]

            elif etype.startswith('INV_Event2Entity'):
                x, pos, neg = emb['entity'][x_id], emb['event'][pos_id], emb['event'][neg_id]

            elif etype.startswith('Entity2Entity'):
                x, pos, neg = emb['entity'][x_id], emb['entity'][pos_id], emb['entity'][neg_id]

            elif etype.startswith('Event2Event'):
                x, pos, neg = emb['event'][x_id], emb['event'][pos_id], emb['event'][neg_id]

            ### get loss ###
            arc_loss.append(self.classifiers[etype](x,pos,neg))

        return arc_loss

    def presample(self, G, etype):
        '''

        :param G:
        :param etype:
        :return: node id - (x, pos, neg) three tensor
        '''
        x, pos = G.all_edges(form='uv', etype=etype, order='eid')

        if etype.startswith('Event2Entity') or etype.startswith('Entity2Entity'):
            cnt = G.number_of_nodes('entity')
        else:
            cnt = G.number_of_nodes('event')
        neg = np.random.choice(cnt, pos.shape[0], True)
        return (x,pos,neg)




class ARC(nn.Module):
    # ARC = adversarial relation classifier

    def __init__(self, nemb=256):
        super().__init__()
        self.discrminator = nn.Bilinear(nemb,nemb,1)
        self.loss = nn.BCEWithLogitsLoss()
        # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.

    def forward(self, x, pos, neg):
        # x = [num_nodes=N, emb_dim=512]
        # pos = [num_nodes=N, emb_dim=512]
        # neg = [num_nodes=N, emb_dim=512]

        # discriminate [N,256]*W*[N,256] = [N,1]

        d1 = self.discrminator(x, pos)
        d2 = self.discrminator(x, neg)
        y = torch.cat((d1.squeeze(1), d2.squeeze(1)), 0)
        # y = logits = [num_nodes=N1+N2]
        label = torch.cat((torch.ones(x.shape[0]), torch.zeros(x.shape[0])), 0)

        if torch.cuda.is_available():
            label = label.cuda()
        loss = self.loss(y, label)

        return loss
