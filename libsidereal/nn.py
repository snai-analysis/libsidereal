from operator import itemgetter
from typing import Protocol, Iterable

import torch
from torch import Tensor

from phytorchx import broadcast_cat

from .simulators.abcsim import SurveySimulator


class AdditionalProtocol(Protocol):
    additional: Tensor


class AddAdditional(torch.nn.Module):
    additional: Tensor

    def __init__(self, sim: AdditionalProtocol):
        super().__init__()
        self.register_buffer('additional', sim.additional)

    def forward(self, t: Tensor):
        return broadcast_cat((t.unsqueeze(-1), self.additional), dim=-1)


class AddExtra(torch.nn.Module):
    extra: Tensor

    def __init__(self, sim: SurveySimulator, extra_keys: Iterable[str] = ('z', 'Av_MW')):
        super().__init__()
        self.register_buffer('extra', torch.cat([
            torch.tensor([i, *itemgetter(*extra_keys)(extra)]).expand(len(sd.field.times), -1)
            for i, (snid, sd, extra) in enumerate(sim)
        ], -2))

    def forward(self, t: Tensor):
        return broadcast_cat((t, self.extra), dim=-1)


class AddSNExtra(torch.nn.Module):
    extra: Tensor

    def __init__(self, sim: SurveySimulator, extra_keys: Iterable[str] = ('z', 'Av_MW')):
        super().__init__()
        self.register_buffer('extra', torch.tensor([
            itemgetter(*extra_keys)(extra)
            for snid, sd, extra in sim
        ]))

    def forward(self, t):
        return broadcast_cat((t, self.extra), dim=-1)


class LogStuff(torch.nn.Module):
    def __init__(self, idx=(0, 3, 5)):
        super().__init__()
        self.idx = idx

    def forward(self, t):
        t[..., self.idx] = t[..., self.idx].asinh()
        return t


class EmbedBand(torch.nn.Module):
    def __init__(self, embedding_dim, n=4, i=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(n, embedding_dim)
        self.i = i

    def forward(self, t):
        return torch.cat((
            t,
            self.embedding(t[..., self.i].to(torch.int))
        ), -1)
