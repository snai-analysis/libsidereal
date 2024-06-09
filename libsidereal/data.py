import gzip
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, Generic, Mapping, TypeVar, Sequence, Any, Union

import astropy.cosmology
import astropy.units
import torch
from astropy.units import Quantity
from pandas import DataFrame
from tqdm.auto import tqdm
from typing_extensions import TypeAlias

from phytorch.quantities import GenericQuantity
from phytorch.units.astro import pc
from slicsim.bandpasses.bandpass import Bandpass
from slicsim.bandpasses.magsys import MagSys
from slicsim.survey import SurveyData

_KT = TypeVar('_KT')


class ExtraData(TypedDict, total=False):
    Av_MW: float
    z: float
    z_cosmo: float

    mu: astropy.units.Quantity
    distance: GenericQuantity


@dataclass
class Datar(Generic[_KT]):
    head: DataFrame = field(repr=False)
    phot: Mapping[_KT, DataFrame] = field(repr=False)

    bandmap: Mapping[str, Bandpass] = field(repr=False)
    magsys: MagSys = field(repr=False)

    sds_name: Path = None
    extras_name: Path = None

    cosmo = astropy.cosmology.FlatLambdaCDM(H0=73.24, Om0=0.28)

    _sds = _extras = None

    @classmethod
    def from_snana(cls, survey: str, root: Path, bands: Mapping[str, Bandpass], magsys: MagSys = None, survey_out: str = None):
        import sncosmo

        data_root = Path(os.environ['SNDATA_ROOT']) / 'lcmerge' / survey

        head, phot = {}, {}

        for fname in tqdm((data_root / data_root.name).with_suffix('.LIST').open().readlines(), leave=False):
            meta, t = sncosmo.read_snana_ascii(gzip.open(data_root / (fname.strip('\n') + '.gz'), 'rt'), 'OBS')
            snid = meta.pop('SNID')
            del meta['END']
            head[snid], phot[snid] = meta, t['OBS'].to_pandas().convert_dtypes()

        head = DataFrame.from_dict(head, orient='index').drop(columns=['SURVEY', 'FILTERS']).convert_dtypes()

        survey_out = survey_out or survey
        survey_root = root / 'surveys' / survey_out
        return cls(head, phot, bands, magsys, survey_root / f'{survey_out}-sds.pt', survey_root / f'{survey_out}-extras.pt')

    @property
    def sds(self) -> Mapping[_KT, SurveyData]:
        if self._sds is None:
            self._sds = {
                snid: SurveyData.from_phot(self.phot[snid], row, self.bandmap, self.magsys)
                for snid, row in self.head.iterrows()
            }

            for sd in self._sds.values():
                sd.field.cache()
        return self._sds

    @property
    def extras(self) -> Mapping[_KT, ExtraData]:
        if self._extras is None:
            self._extras = DataFrame.from_dict({
                'Av_MW': (self.head['MWEBV'] / 3.1).to_dict(),
                'z': (z := self.head['REDSHIFT_FINAL'].to_dict()),
                'z_cosmo': (z_cosmo := z),
                'mu': {key: self.cosmo.distmod(_z) for key, _z in z_cosmo.items()}
            }).T.to_dict()

            for extra in self._extras.values():
                extra['distance'] = 10**(extra['mu'].value/5) / (1 + extra['z']) * 10*pc
        return self._extras

    def save(self):
        self.sds_name.parent.mkdir(exist_ok=True)
        torch.save(self.sds, self.sds_name)

        self.extras_name.parent.mkdir(exist_ok=True)
        torch.save(self.extras, self.extras_name)


# @dataclass
# class Datar(AbstractDatar[_KT], Generic[_KT]):
#     survey: str
#     root: Path = field(repr=False)
#     bands: Mapping[str, Bandpass] = field(repr=False)
#
#     cosmo = astropy.cosmology.FlatLambdaCDM(H0=73.24, Om0=0.28)
#
#     _head = _phot = _sds = _extras = None
#
#     def __post_init__(self):
#         self.snana_root = Path(os.environ['SNDATA_ROOT'])
#         self.data_root = self.snana_root / 'lcmerge' / self.survey
#         self.survey_root = self.root / 'surveys' / self.survey
#
#         self.sds_name = self.survey_root / f'{self.survey}-sds.pt'
#         self.extras_name = self.survey_root / f'{self.survey}-extras.pt'
#
#     def _get_head_phot(self):
#         import sncosmo
#
#         _head, self._phot = {}, {}
#
#         for fname in tqdm((self.data_root / self.data_root.name).with_suffix('.LIST').open().readlines(), leave=False):
#             meta, t = sncosmo.read_snana_ascii(gzip.open(self.data_root / (fname.strip('\n') + '.gz'), 'rt'), 'OBS')
#             snid = meta.pop('SNID')
#             del meta['END']
#             _head[snid], self._phot[snid] = meta, t['OBS'].to_pandas().convert_dtypes()
#
#         self._head = DataFrame.from_dict(_head, orient='index').drop(columns=['SURVEY', 'FILTERS']).convert_dtypes()
#
#     @property
#     def head(self) -> DataFrame:
#         if self._head is None:
#             self._get_head_phot()
#         return self._head
#
#     @property
#     def phot(self) -> Mapping[_KT, DataFrame]:
#         if self._phot is None:
#             self._get_head_phot()
#         return self._phot


class BayeSNData_SN(TypedDict, total=False):
    t0: float
    z: float
    Av_MW: float

    mask: Sequence[bool]

    times: Sequence[float]
    bands: Sequence[str]

    fluxcalerr: Sequence[float]

    zp_mag_mean: Sequence[float]
    zp_mag_std: Sequence[float]
    noise: Sequence[float]
    sky: Sequence[float]
    area: Sequence[float]
    gain: Sequence[float]

    fluxcal: Sequence[float]
    fluxcal_nonoise: Sequence[float]

    del_M: float
    Av: float
    Rv: float
    theta: float
    eps: Any

    mu: Union[Quantity, float]


BayeSNData: TypeAlias = Mapping[Any, BayeSNData_SN]
