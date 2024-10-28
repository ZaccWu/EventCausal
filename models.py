import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Any, Optional, Tuple

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GRL_Layer(torch.nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()
    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class CasualTPP(nn.Module):
    def __init__(self, in_dim, h_dim=32, n_layers=2):
        super(CasualTPP, self).__init__()
        self.LSTM = nn.LSTM(in_dim, h_dim, n_layers, batch_first=True, bidirectional=False)
        self.treat_ll = nn.Linear(h_dim, 2)
        self.out_ll = nn.Linear(h_dim, 1)
        self.act = nn.LeakyReLU()
        self.grl = GRL_Layer()

    def forward(self, feature):
        ht, _ = self.LSTM(feature) # ht: (batch*num_sample, K, h_dim)
        # predict treatment with gradient reversal
        treat_pred = self.act(self.treat_ll(self.grl(ht[:, -1, :])))  # treat_pred: (batch*num_stock, 2)
        # predict outcome with gradient reversal
        out_pred = self.act(self.out_ll(ht[:, -1, :]))  # out_pred: (batch*num_stock, 1)
        return treat_pred, out_pred