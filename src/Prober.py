
import torch
import torch.nn as nn


class LinearProber(nn.Module):
    def __init__(self, nfeat, nclass, nemb=512, dropout=0.1, task='event'):
        super().__init__()

        self.nfeat = nfeat
        self.project = nfeat != nemb
        if self.project:
            self.emb = nn.Linear(nfeat,nemb)

        self.node = ( task == 'event' or task == 'entity')
        if self.node: # node classification
            self.mlp = nn.Linear(nemb, nclass)

        else: # edge classification
            self.mlp = nn.Linear(nemb * 2, nclass)

    def forward(self, x):
        # x = [batch_size=32, nfeat=768*2]
        if self.project:
            if self.node:
                x = self.emb(x)
            else:
                x = torch.cat([self.emb(x[:,:self.nfeat]), self.emb(x[:,self.nfeat:])],dim=1)
        # x [batch_size=32, nemb=256*2]
        x = self.mlp(x)


        # x = torch.cat([mlp(x).unsqueeze(1) for mlp in self.mlps], dim=1) #used in binary prober
        # # x = [batch_size=32, nclass=823, 2]

        # x [batch_size=32, nclass=36]
        return x



class noProjectProber(nn.Module):
    def __init__(self, nfeat, nclass, dropout=0.1):
        super().__init__()
        self.mlp = MLP(nfeat=nfeat, nclass=nclass, dropout=dropout)
    def forward(self, x):
        # x = [batch_size=32, nfeat=768*2]
        return self.mlp(x)

class ProjectedProber(nn.Module):
    def __init__(self, nfeat, nclass, input_emb=256, dropout=0.1):
        super().__init__()
        self.mlp = MLP(nfeat=nfeat, nclass=nclass, dropout=dropout)
        self.if_project = (nfeat != input_emb)
        if self.if_project:
            self.proj = nn.Linear(input_emb, nfeat)
    def forward(self, x):
        # x = [batch_size=32, nfeat=768*2]
        if self.if_project:
            x = self.proj(x)
        return self.mlp(x)



class Prober(nn.Module):
    def __init__(self, nfeat, nclass, nemb=512, dropout=0.1, task='event'):
        super().__init__()

        self.nfeat = nfeat
        self.project = nfeat != nemb
        if self.project:
            self.emb = nn.Linear(nfeat,nemb)

        self.node = ( task == 'node')
        if self.node: # node classification
            self.mlp = MLP(nfeat=nemb, nclass=nclass, dropout=dropout)

        else: # edge classification
            self.mlp = MLP(nfeat=nemb * 2, nclass=nclass, dropout=dropout)

    def forward(self, x):
        # x = [batch_size=32, nfeat=768*2]
        if self.project:
            if self.node:
                x = self.emb(x)
            else:
                x = torch.cat([self.emb(x[:,:self.nfeat]), self.emb(x[:,self.nfeat:])],dim=1)
        # x [batch_size=32, nemb=256*2]
        x = self.mlp(x)


        # x = torch.cat([mlp(x).unsqueeze(1) for mlp in self.mlps], dim=1) #used in binary prober
        # # x = [batch_size=32, nclass=823, 2]

        # x [batch_size=32, nclass=36]
        return x


class MLP(nn.Module):
    def __init__(self, nfeat=256, nclass=2, dropout=0.1):
        super().__init__()
        nhid = nfeat // 2
        self.f = nn.Sequential(
            nn.Linear(nfeat, nhid),
            nn.Tanh(),
            nn.LayerNorm(nhid),
            nn.Dropout(dropout),
            nn.Linear(nhid, nclass)
        )

    def forward(self, x):
        return self.f(x)
