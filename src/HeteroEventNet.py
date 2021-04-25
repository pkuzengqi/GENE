
import torch
import torch.nn as nn
from utils import get_metapath_dict

from DGI import DGI_trainer
from SEM import SEM
from ARC import ARC_trainer
from SKG import SKG_trainer
from HeteroRGCN import HeteroRGCN




class HeteroEventNet(nn.Module):
    def __init__(self, G, valG, nfeat=768, nemb=768, nhid=256, dropout=0.1, pooling= 'cat', max_batch=1000, views=None, batch_size=16, model_base='SEM_ARC'):
        '''
        :param G:
        :param pooling:
            'cat': concate three embedding results from views, real output nemb would be
            'avg': average three embedding results from views
        :return:
        '''
        super().__init__()
        self.nemb = nemb
        self.pooling = pooling
        self.metapath_dict = get_metapath_dict(G.etypes)
        self.max_batch = max_batch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        if views == None or views == '':
            self.views = ['event2entity','entity2entity','event2event','all']
        else:
            self.views = views.split()

        # graph encoder
        if pooling == 'cat':
            self.encoder = HeteroRGCN(G,in_size=nfeat, hidden_size=nemb//3, out_size=nemb//3).to(self.device)
        else:
            self.encoder = HeteroRGCN(G, in_size=nfeat, hidden_size=nemb, out_size=nemb).to(self.device)
        if pooling == 'weighted':
            self.w0 = nn.Linear(nemb,nemb)
            self.w1 = nn.Linear(nemb,nemb)
            self.w2 = nn.Linear(nemb, nemb)

        self.if_SEM = 'SEM' in model_base
        self.if_DGI = 'DGI' in model_base
        self.if_SKG = 'SKG' in model_base
        self.if_ARC = 'ARC' in model_base

        if self.if_SEM:
            self.SEM = SEM(nfeat=nfeat, nemb=nemb, nhid=nemb, dropout=dropout)
        if self.if_DGI:
            self.DGI = DGI_trainer(G, valG, nemb=nemb,batch_size=batch_size)
        if self.if_SKG:
            self.SKG = SKG_trainer(G, valG, nemb=nemb, batch_size=batch_size)
        if self.if_ARC:
            self.ARC = ARC_trainer(G, valG, nemb=nemb,batch_size=batch_size)

    def resample(self, G, valG):
        if self.if_DGI:
            self.DGI.sample(G,valG)
        if self.if_SKG:
            self.SKG.sample(G,valG)
        if self.if_ARC:
            self.ARC.sample(G,valG)

    def forward(self, G, homoG, max_batch=None, metapath_dict=None, batch_num =0):



        # max_batch: the largest number to sample subgraphs to get loss
        # by default max_batch=self.args.max_batch_for_train
        # if given, change to self.args.max_batch_for_eval
        if max_batch == None:
            max_batch = self.max_batch

        if metapath_dict == None:
            metapath_dict = self.metapath_dict

        #################### RGCN Encoder ####################
        # encode nodes with RGCN
        emb = self.encode(G, metapath_dict)  # emb = {'event': [4353,768], 'entity':[3688,768]}

        #################### SEM Loss ####################
        # SEM loss evaluates the similarity between node attributes and embed results

        sem_loss = [torch.tensor(0).float().to(self.device)]
        if self.if_SEM:
            sem_loss = [self.SEM(G.nodes[ntype].data['x'], emb[ntype]) for ntype in G.ntypes]
        sem_loss = torch.stack(sem_loss).mean()


        #################### DGI Loss ####################
        dgi_loss = [torch.tensor(0).float().to(self.device)]
        if self.if_DGI:
            dgi_loss = self.DGI(G, emb, batch_num)
        dgi_loss = torch.stack(dgi_loss).mean()


        #################### SKG Loss ####################
        skg_loss = [torch.tensor(0).float().to(self.device)]
        if self.if_SKG:
            skg_loss = self.SKG(G, emb, batch_num)
        skg_loss = torch.stack(skg_loss).mean()


        #################### ARC Loss ####################
        arc_loss = [torch.tensor(0).float().to(self.device)]
        if self.if_ARC:
            arc_loss = self.ARC(G, emb, batch_num)
        arc_loss = torch.stack(arc_loss).mean()


        return sem_loss,dgi_loss,skg_loss, arc_loss


    def encode(self, G, metapath_dict = None):

        # in eval mode, need to specify new metapath_dict
        if metapath_dict == None:
            metapath_dict = self.metapath_dict

        # initiate emb
        emb = {ntype: [] for ntype in G.ntypes}


        # get graph view and go through encoder

        for view in self.views:
            if view == 'all':
                g = G.to(self.device)
            else:
                # g = dgl.transform.metapath_reachable_graph(G, self.metapath_dict[view]).to(self.device)
                g = G.edge_type_subgraph(metapath_dict[view]).to(self.device)



            h_dict = self.encoder(g) # {ntype: [emb]}
            for ntype, h in h_dict.items():
                emb[ntype].append(h)

        #################### graph sampler ####################
        # result aggregation: concatenation or average or weighted sum
        # cat: emb = {'event': [3*[N,256]], 'entity':[3*[N,256]]}
        # avg: emb = {'event': [3*[N,768]], 'entity':[3*[N,768]]}
        # no: emb = {'event': [N,768], 'entity':[N,768]}
        if self.pooling == 'cat':
            for ntype, h in emb.items():
                emb[ntype] = torch.cat(emb[ntype], dim=1)
        elif self.pooling == 'avg':
            for ntype, h in emb.items():
                emb[ntype] = torch.mean(torch.cat([e.unsqueeze(0) for e in h], dim=0), dim=0)
        elif self.pooling == 'weighted':
            for ntype, h in emb.items():
                emb[ntype] = torch.mean(torch.cat([self.w0(h[0]).unsqueeze(0),self.w1(h[1]).unsqueeze(0),self.w2(h[2]).unsqueeze(0)], dim=0), dim=0)
        elif self.pooling == 'no':
            for ntype, h in emb.items():
                if len(h) > 0:
                    emb[ntype] = h[0]
        # emb = {'event': [4353,768], 'entity':[3688,768]}
        return emb


