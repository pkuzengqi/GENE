import torch
import torch.nn as nn
import dgl
from utils import vocab_size_dict, construct_homo_from_hetero_dglgraph


class DGI_trainer(nn.Module):
    '''
    presampled_batched version, used to compare with arc
    '''

    def __init__(self,  G, valG, nemb=256, batch_size=16):
        super().__init__()
        self.nemb = nemb
        self.batch_size = batch_size

        self.discrminator = nn.Bilinear(nemb,nemb,1)
        self.sigm = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        self.sample(G,valG)

    def sample(self,G,valG):
        homo_g = construct_homo_from_hetero_dglgraph(G)
        self.samples = self.presample(G,homo_g)
        homo_valg = construct_homo_from_hetero_dglgraph(valG)
        self.val_samples = self.presample(valG,homo_valg)



    def get_loss(self, x1, x2):
        # x1 = [num_nodes=N, emb_dim=512]

        # readout summary
        s = self.sigm(torch.mean(x1,0))
        # s = [emb_dim=512]

        ######### discriminate over all x1 ##########
        # discriminate [N,256]*W*[N,256] = [N,1]
        d1 = self.discrminator(x1,s.unsqueeze(0).repeat((x1.size()[0],1)))
        d2 = self.discrminator(x2,s.unsqueeze(0).repeat((x2.size()[0],1)))
        y = torch.cat((d1.squeeze(1), d2.squeeze(1)), 0)
        # y = logits = [num_nodes=N1+N2]
        label = torch.cat((torch.ones(x1.shape[0]), torch.zeros(x2.shape[0])), 0)


        if torch.cuda.is_available():
            label = label.cuda()
        loss = self.loss(y, label)

        return loss

    def presample(self, G, homoG):
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
                                                       num_hops=2,
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

        dgi_loss = []
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
            loss = self.get_loss(pos,neg)
            dgi_loss.append(loss)


        return dgi_loss





class DGI(nn.Module):
    def __init__(self, nemb=256, nheads=2, nlayers=3, nhid=256, dropout=0.1):
        super().__init__()



        self.discrminator = nn.Bilinear(nemb,nemb,1)

        self.sigm = nn.Sigmoid()

        self.loss = nn.BCEWithLogitsLoss()
        # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.

    def forward(self, x1, x2):
        # x1 = [num_nodes=N, emb_dim=512]



        # readout summary
        s = self.sigm(torch.mean(x1,0))
        # s = [emb_dim=512]

        ######### discriminate over all x1 ##########
        # discriminate [N,256]*W*[N,256] = [N,1]
        d1 = self.discrminator(x1,s.unsqueeze(0).repeat((x1.size()[0],1)))
        d2 = self.discrminator(x2,s.unsqueeze(0).repeat((x2.size()[0],1)))
        y = torch.cat((d1.squeeze(1), d2.squeeze(1)), 0)
        # y = logits = [num_nodes=N1+N2]
        label = torch.cat((torch.ones(x1.shape[0]), torch.zeros(x2.shape[0])), 0)

        ######### discriminate over only s1 and s2 ##########
        # d1 = self.discrminator(ys1, s)
        # d2 = self.discrminator(ys2, s)
        # y = torch.cat((d1, d2), 0) # [2]
        # label = torch.tensor([1,0]).float()


        if torch.cuda.is_available():
            label = label.cuda()
        loss = self.loss(y, label)

        return loss
