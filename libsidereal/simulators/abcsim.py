from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from numbers import Number
from operator import itemgetter
from typing import Generic, TypeVar, Mapping, Iterator, Sequence, ClassVar, Iterable

import pyro.distributions
import torch
from more_itertools import unique_everseen
from torch import Tensor

from slicsim.bandpasses.bandpass import Bandpass
from slicsim.survey import SurveyData, ExtraData

_KT = TypeVar('_KT')
_OT = TypeVar('_OT')


@dataclass
class SurveySimulator(Generic[_KT]):
    sds: Mapping[_KT, SurveyData]
    extras: Mapping[_KT, ExtraData]

    def __iter__(self) -> Iterator[tuple[_KT, SurveyData, ExtraData]]:
        for snid, sd in self.sds.items():
            yield snid, sd, self.extras[snid]

    @cached_property
    def bandmap(self) -> Mapping[Bandpass, Number]:
        return dict(map(reversed, enumerate(
            unique_everseen(chain.from_iterable(
                sd.field.bands for snid, sd, extra in self
            ))
        )))

    @cached_property
    def lens(self) -> tuple[int, ...]:
        return tuple(len(sd.field.times) for snid, sd, extra in self)


@dataclass
class ConcatenatedSurveySimulator(SurveySimulator[_KT], Generic[_KT]):
    keys: Sequence[_KT]

    def __iter__(self) -> Iterator[tuple[_KT, SurveyData, ExtraData]]:
        for snid in self.keys:
            yield snid, self.sds[snid], self.extras[snid]

    @cached_property
    def sn_extra(self) -> Tensor:
        return torch.tensor([[extra['z'], extra['Av_MW']] for snid, sd, extra in self])


class ConcatenatedDatasetSurveySimulator(ConcatenatedSurveySimulator[_KT], Generic[_KT], ABC):
    @property
    @abstractmethod
    def additional(self) -> Tensor: ...


class ConcatenatedDatasetFluxcalSurveySimulator(ConcatenatedDatasetSurveySimulator[_KT], Generic[_KT]):
    @cached_property
    def additional(self) -> Tensor:
        return torch.concatenate(tuple(
            torch.stack((
                _times := torch.as_tensor(sd.field.times).to(sd.fluxcalerr),
                sd.fluxcalerr.new_tensor(
                    tuple(map(self.bandmap.__getitem__, sd.field.bands))),
                sd.fluxcalerr
            ), dim=-1)
            for snid, sd, extra in self
        ), dim=-2)


class LightcurveSimulator(SurveySimulator[_KT], Generic[_KT, _OT], ABC):
    def postprocess(self, lcs: Mapping[_KT, Tensor]) -> _OT:
        raise NotImplementedError

    @abstractmethod
    def simulate_one(self, snid: _KT, *latent_args, **global_kwargs) -> pyro.distributions.Distribution: ...


    data_ndim: ClassVar[int] = 1
    latent_names: ClassVar[Iterable[str]] = ()

    def __call__(self, **kwargs) -> _OT:
        return self.postprocess({
            snid: pyro.sample(
                f'data_{snid}',
                self.simulate_one(snid, *latents, **{key: val for key, val in kwargs.items() if key not in self.latent_names})
            )
            for (snid, sd, extra), *latents in zip(
                self, *(kwargs[name].unbind(-1) for name in self.latent_names)
            )
        })


class ConcatenatedLightcurveSimulator(ConcatenatedSurveySimulator[_KT], LightcurveSimulator[_KT, Tensor], Generic[_KT], ABC):
    def postprocess(self, lcs: Mapping[_KT, Tensor]) -> Tensor:
        return torch.cat(itemgetter(*self.keys)(lcs), dim=-1)

    def unpostprocess(self, res: Tensor) -> Mapping[_KT, Tensor]:
        return dict(zip(self.keys, res.split(self.lens, dim=-1)))
