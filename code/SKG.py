import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from utils import vocab_size_dict, construct_homo_from_hetero_dglgraph

class SKG_trainer(nn.Module):
    '''
    presampled_batched version, used to compare with arc
    '''

    def __init__(self, G, valG, nemb=256, batch_size=16):
        super().__init__()
        self.nemb = nemb
        self.batch_size = batch_size

        self.sample(G,valG)

    def sample(self,G,valG):
        homo_g = construct_homo_from_hetero_dglgraph(G)
        self.samples = self.presample(homo_g)
        homo_valg = construct_homo_from_hetero_dglgraph(valG)
        self.val_samples = self.presample(homo_valg)

    def presample(self, homoG):
        '''

        :param G: homo g
        :return: node id - (x, pos, neg) three tensor
        '''

        seed = []
        pos = []
        neg = []

        cur = 0
        N = 3000
        for nf in dgl.contrib.sampling.NeighborSampler(g=homoG,
                                                       batch_size=1,
                                                       expand_factor=10,
                                                       num_hops=1,
                                                       shuffle=True):
            if cur > N:
                break
            cur += 1

            # turn nodeflow to subgraph, get x and adj
            nf_nodes = torch.cat([nf.layer_parent_nid(lid) for lid in range(nf.num_layers)], dim=0)
            subg2 = homoG.subgraph(nf_nodes)
            x = subg2.parent_nid
            # s = subg2.map_to_subgraph_nid(nf.layer_parent_nid(nf.num_layers - 1))
            s = nf.layer_parent_nid(nf.num_layers - 1)
            if len(neg) == 0:
                neg.append(x)
            else:
                seed.append(s)
                pos.append(x)
                neg.append(x)


        return (seed,pos,neg[:-1])



    def forward(self, G, emb, batch_num=0):
        # input emb = {'event': [3721,768], 'entity':[3688,768]}
        homo_emb = torch.cat([emb[ntype] for ntype in G.ntypes], dim=0)

        skg_loss = []
        if emb['event'].shape[0] == vocab_size_dict['train']['event']:
            samples = self.samples
        else:
            samples = self.val_samples



        ### get pre-sampled results ###
        (x_id, pos_id, neg_id) = samples

        ### get batches ###
        st = batch_num*self.batch_size % len(x_id)
        en = (batch_num+1)*self.batch_size % len(x_id)
        if en == st == 0 or en < st:
            en = len(x_id)

        for idx in range(st,en):
            x, pos, neg = homo_emb[x_id[idx]], homo_emb[pos_id[idx]], homo_emb[neg_id[idx]]

            ### get loss ###
            loss = self.get_loss(pos,x,neg)
            skg_loss.append(loss)


        return skg_loss


    def get_loss(self, v_pos, u_pos, v_neg):

        # x1 = v_pos = [num_nodes=N, emb_dim=512]
        # s1 = u_pos = [1,emb=512]
        # x2 = v_neg = [num_nodes=M, emb_dim=512]

        pos_score = torch.mul(u_pos.repeat((v_pos.shape[0], 1)), v_pos)
        pos_score = torch.sum(pos_score, dim=1)
        log_target = F.logsigmoid(pos_score).squeeze().mean()

        neg_score = torch.mul(u_pos.repeat((v_neg.shape[0], 1)), v_neg)
        neg_score = torch.sum(neg_score, dim=1)
        sum_log_sampled = F.logsigmoid(-1 * neg_score).squeeze().mean()

        loss = log_target + sum_log_sampled

        return -1 * loss.sum()

