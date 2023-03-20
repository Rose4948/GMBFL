import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim):
        super(Propogator, self).__init__()


        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Tanh()
        )

    def forward(self,state, state_cur, A):
        a_t= torch.bmm(A, state)
        a = torch.cat((a_t, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_t, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGANN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim,nheads):
        super(GGANN, self).__init__()
        self.state_dim = state_dim
        self.dropout=0.1
        self.alpha=0.2
        nhid = int(state_dim / nheads)
        self.linear= nn.Linear(self.state_dim, self.state_dim)

        # Propogation Model
        self.propogator = Propogator(self.state_dim)

        # Output Model
        self.out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )

        self._initialization()
        self.attentions = [GraphAttentionLayer(state_dim, nhid, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state,left, A):
        prop_state = F.dropout(prop_state, self.dropout, training=self.training)
        prop_state = torch.cat([att(prop_state, A) for att in self.attentions], dim=-1)
        prop_state = self.propogator(prop_state,prop_state, A )
        return prop_state

class SpGGANN(nn.Module):
    """
    Gated Graph Attention Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim,nheads):
        super(SpGGANN, self).__init__()
        self.state_dim = state_dim
        self.dropout=0.1
        self.alpha=0.2
        nhid = int(state_dim / nheads)
        self.linear= nn.Linear(self.state_dim, self.state_dim)

        # Propogation Model
        self.propogator = Propogator(self.state_dim)

        # Output Model
        self.out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )
        # self.self_att1 = selfattention( 4, self.state_dim )
        # self.self_att0 = selfattention( 28, self.state_dim, flag=True )
        self._initialization()
        self.attentions = [SpGraphAttentionLayer(state_dim, nhid, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state,left, A):
        prop_state = F.dropout(prop_state, self.dropout, training=self.training)
        prop_state = torch.cat([att(prop_state, A) for att in self.attentions], dim=-1)
        prop_state = self.propogator(prop_state,prop_state, A )
        return prop_state

