import torch.nn as nn


class SEM(nn.Module):
    def __init__(self, nfeat=768, nemb=512, nhid=512, dropout=0.1):
        super().__init__()

        self.f = nn.Sequential(
            nn.Linear(nemb, nhid),
            nn.Tanh(),
            nn.LayerNorm(nhid),
            nn.Dropout(dropout),
            nn.Linear(nhid, nfeat)
        )

        self.loss = nn.MSELoss()


    def forward(self, x, y):
        # x = [num_nodes=N, nfeat=768]
        # y = [num_nodes=N, nemb=512]
        return self.loss(x,self.f(y))

