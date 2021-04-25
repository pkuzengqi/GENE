import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


'''
modified from 
https://docs.dgl.ai/tutorials/hetero/1_basics.html
https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/4_rgcn.html
'''

class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size=768, hidden_size=512, out_size=512, num_bases=-1):
        super(HeteroRGCN, self).__init__()


        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes, num_bases)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes, num_bases)

        print_param_num = False
        if print_param_num:
            print('Parameters:')
            for p in self.parameters():
                if p.requires_grad:
                    print(p.name, p.numel())
            print()


    def forward(self, G):
        # use data[x] as input feature
        embed_dict = {ntype: Variable(G.nodes[ntype].data['x'], requires_grad=False) for ntype in G.ntypes}
        h_dict = self.layer1(G, embed_dict)
        h_dict = self.layer2(G, h_dict)

        return h_dict # return a dict of {ntype: [emb]}


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes, num_bases=-1):
        super(HeteroRGCNLayer, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.stat_weight = nn.Linear(self.in_size,self.out_size)
        self.num_rels = len(etypes)

        # uncomment this IF to perform weight decomposition
        if num_bases < 0:
            num_bases = self.num_rels // 3
        self.num_bases = num_bases


        if self.num_bases < self.num_rels  and self.num_bases > 0:


            self.edge_map = dict()
            for i,name in enumerate(etypes):
                self.edge_map[name]=torch.tensor(i, requires_grad=False).to(self.device) # etype_name to index

            self.base_weight = nn.ModuleList([nn.Linear(in_size, out_size) for _ in range(self.num_bases)])
            self.edge_decompose = nn.ModuleDict({
                name: torch.nn.Conv1d(in_channels=self.num_bases, out_channels=1, kernel_size=1) for name in etypes
            })


        else:
            # W_r for each relation
            self.weight = nn.ModuleDict({
                name: nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):

        funcs = {}

        if self.num_bases > 0:
            Wh_V = dict()
            for srctype in G.ntypes:
                Wh_V[srctype] = torch.cat(
                    [l(feat_dict[srctype]).unsqueeze(1) for i, l in enumerate(self.base_weight)],
                    1)  # after concat [n=4353, num_base=26, out_size=768]


        for srctype, etype, dsttype in G.canonical_etypes:
            if self.num_bases > 0:
                #### a solution with conv1d
                Wh = self.edge_decompose[etype](Wh_V[srctype]).squeeze(1)

            else:
                # Compute W_r * h
                # h dim=[N, in_size]
                Wh = self.weight[etype](feat_dict[srctype])

            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))



        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')

        # return the updated node feature dictionary
        return {ntype : F.leaky_relu(G.nodes[ntype].data['h'] + self.stat_weight(feat_dict[ntype])) for ntype in G.ntypes}


