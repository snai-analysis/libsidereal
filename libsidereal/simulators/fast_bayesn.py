from dataclasses import dataclass, field
from math import pi
from typing import Mapping, TypeVar, Sequence, cast

import torch
from torch import Tensor, LongTensor

from phytorch.constants import h as planck_constant, c as speed_of_light
from phytorch.interpolate import Linear1dInterpolator
from phytorch.quantities import Quantity
from phytorch.units import Unit
from phytorch.units.astro import parsec
from phytorch.units.si import angstrom
from phytorchx import mid_one
from slicsim.bandpasses.bandpass import Bandpass
from slicsim.extinction import Fitzpatrick99
from slicsim.sources.bayesn import TrainedBayeSNSource
from slicsim.survey import SurveyData, ExtraData

from .abcsim import ConcatenatedDatasetFluxcalSurveySimulator, _KT

_TT = TypeVar('_TT', bound=Tensor)


class FastBayeSN:
    @classmethod
    def from_sds(cls, source: TrainedBayeSNSource, sds: Mapping[_KT, SurveyData], extras: Mapping[_KT, ExtraData], snids: Sequence[_KT] = None):
        if snids is None:
            snids = sds.keys()

        return cls(
            source=source,
            times=torch.cat([torch.tensor(sds[snid].field.times) for snid in snids]),
            bands=tuple(band for snid in snids for band in sds[snid].field.bands),
            zps=cast(Quantity, torch.stack([
                sds[snid].field.magsys.zp_counts(band)
                for snid in snids
                for band in sds[snid].field.bands
            ])),
            Nobss=cast(LongTensor, torch.tensor([len(sds[snid].fluxcalerr) for snid in snids])),
            zs=torch.tensor([extras[snid]['z'] for snid in snids]),
            Av_MWs=torch.tensor([extras[snid]['Av_MW'] for snid in snids]),
            distances=torch.tensor([extras[snid]['distance'].to(parsec) for snid in snids]) * parsec
        )

    def __init__(
        self,
        source: TrainedBayeSNSource,
        times: Tensor,              # (Nobs,) [day]
        bands: Sequence[Bandpass],  # (Nobs,)
        zps: Quantity,              # (Nobs,)

        Nobss: LongTensor,    # (NSN,)
        zs: Tensor,           # (NSN,)
        Av_MWs: Tensor,       # (NSN,)
        distances: Quantity  # (NSN,)
    ):
        self.source = source
        self.times = times

        self.snobsidx = torch.arange(len(Nobss)).repeat_interleave(Nobss)

        self.scale_factors = 1 / (1 + self.NSN_to_Nobs(zs))

        wave_extent = self.source.bayesn_wave[(0, -1),]

        # noinspection PyTypeChecker
        wave_mask = torch.logical_and(
            wave_extent[0] <= self.source.grid_wave,
            self.source.grid_wave <= wave_extent[1])

        self.grid_wave = mid_one(self.source.grid_wave[wave_mask], -1)
        dgrid_wave = self.source.grid_wave[wave_mask].diff(n=1, dim=-1)
        grid_flux_Hsiao = mid_one(self.source.grid_flux[..., wave_mask], -1)

        grid_counts_Hsiao = (
            grid_flux_Hsiao * self.source.flux_unit * dgrid_wave * angstrom
            / (planck_constant * speed_of_light / (self.grid_wave * angstrom))
        ).mT

        # Here "scale_factor" because of time dilation
        self._factor = (self.scale_factors / self.NSN_to_Nobs(4*pi * distances**2) * grid_counts_Hsiao.unit / zps).to(Unit()).value

        self.iHsiao = Linear1dInterpolator(self.source.grid_phase, 10**(27.5/2.5) * grid_counts_Hsiao.value)

        self.bandtrans_mag = torch.stack([
            torch.where(
                torch.logical_and(
                    band.minwave < grid_wave_obs,
                    grid_wave_obs < band.maxwave),
                band.mag(grid_wave_obs).nan_to_num(nan=float('inf')),
                float('inf')
            ) for band, scale_factor in zip(bands, self.scale_factors)
            for grid_wave_obs in [self.grid_wave / scale_factor]
        ])  # (Nobs, Ngrid_wave)

        self.MWext_mag = self.NSN_to_Nobs(
            Av_MWs * Fitzpatrick99(3.1).mag(self.grid_wave.unsqueeze(-1) * (1+zs)),
        ).mT  # (Nobs, Ngrid_wave)

    def NSN_to_Nobs(self, arr: _TT, dim: int = -1) -> _TT:
        # return arr.repeat_interleave(self.Nobss, dim=dim)
        return arr.index_select(dim, self.snobsidx)


    def bandcountscal(
        self,
        delta_ts: Tensor, delta_Ms: Tensor, thetas: Tensor, es: Tensor,
        Avs: Tensor, Rvs: Tensor
    ):
        phases = (self.times - self.NSN_to_Nobs(delta_ts)) * self.scale_factors  # (Nobs,)

        bsn_mag = self.source.bayesn_spline.evaluate(
            self.NSN_to_Nobs(
                self.source.set_params(delta_M=delta_Ms, theta=thetas, e=es).bayesn_grid_mag,
                -3
            ).unsqueeze(-3),
            phases.unsqueeze(-1), self.grid_wave
        )  # (Nobs, Ngrid_wave)

        ext_mag = self.NSN_to_Nobs(
            Avs.unsqueeze(-1) * Fitzpatrick99(Rvs.unsqueeze(-1)).mag(self.grid_wave),
            -2
        )  # (Nobs, Ngrid_wave)

        specs = self.iHsiao(phases) * 10 ** (-0.4 * (
            bsn_mag + ext_mag + self.MWext_mag + self.bandtrans_mag
        ))

        ret = self._factor * specs.sum(-1)

        return ret

    def __call__(self, delta_ts: Tensor, delta_Ms: Tensor, thetas: Tensor, es: Tensor, Avs: Tensor, Rvs: Tensor):
        return self.bandcountscal(delta_ts, delta_Ms, thetas, es.mT, Avs, Rvs)


@dataclass
class FastBayeSNSimulator(ConcatenatedDatasetFluxcalSurveySimulator):
    source: TrainedBayeSNSource
    model: FastBayeSN = field(init=False)

    def __post_init__(self):
        self.model = FastBayeSN.from_sds(self.source, self.sds, self.extras, self.keys)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
