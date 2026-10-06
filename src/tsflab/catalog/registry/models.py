"""Flat, lazy model catalog and model specification contracts."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Literal, Type

from pydantic import BaseModel


@dataclass(frozen=True)
class ModelArtifact:
    """Pinned external artifact required by an optional model runtime path."""

    name: str
    url: str
    revision: str
    sha256: str
    filename: str
    required: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.revision or not self.filename:
            raise ValueError("artifact name, revision, and filename must be non-empty")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError(f"artifact {self.name!r} needs a lowercase SHA-256 digest")
        if not self.url.startswith(("hf://", "https://", "file://")):
            raise ValueError(f"artifact {self.name!r} URL must use hf://, https://, or file://")
        if self.url.startswith("hf://"):
            from tsflab.release.hub.uri import parse

            if parse(self.url).revision != self.revision:
                raise ValueError(
                    f"artifact {self.name!r} hf:// URI must be pinned to its revision"
                )
        elif self.url.startswith("https://") and self.revision not in self.url:
            raise ValueError(
                f"artifact {self.name!r} URL must contain its pinned revision"
            )
        if "/" in self.filename or "\\" in self.filename:
            raise ValueError(f"artifact {self.name!r} filename must be a basename")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    module: str
    model_class: type
    factory: Callable
    params_schema: Type[BaseModel]
    config_path: str = ""
    model_card: str = ""
    smoke_config: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    components: tuple[str, ...] = ()
    contract_task: dict[str, int | str] = field(default_factory=dict)
    contract_seeds: tuple[int, ...] = (0,)
    #: The model needs the dataset adjacency (``adj_mx``, injected by the runner) and
    #: refuses to build without it; the admission contract supplies a synthetic graph.
    requires_graph: bool = False
    artifacts: tuple[ModelArtifact, ...] = ()
    artifact_factory: Callable | None = None
    # Opt-in paper objective used only while training; contract in
    # ``tsflab.experiments.runner.objective``. ``training_setup`` prepares it once
    # from the training split (for example fitting a label basis).
    training_objective: Callable | None = None
    training_setup: Callable | None = None

    def __post_init__(self) -> None:
        output_capabilities = self.capabilities & {
            "quantile-output", "distribution-output"
        }
        if len(output_capabilities) > 1:
            raise ValueError(f"model {self.name!r} declares conflicting output capabilities")
        if not self.task_modes:
            raise ValueError(f"model {self.name!r} declares no supported task mode")
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"model {self.name!r} declares duplicate components")
        if self.training_objective is not None and not callable(self.training_objective):
            raise TypeError(f"model {self.name!r} training_objective must be callable")
        if self.training_setup is not None:
            if not callable(self.training_setup):
                raise TypeError(f"model {self.name!r} training_setup must be callable")
            if self.training_objective is None:
                raise ValueError(
                    f"model {self.name!r} declares training_setup without training_objective"
                )
        if "inference-only" in self.capabilities:
            if "pretraining-stage" in self.capabilities:
                raise ValueError(
                    f"model {self.name!r} cannot be inference-only and pretraining-stage"
                )
            if self.training_objective is not None:
                raise ValueError(
                    f"model {self.name!r} cannot be inference-only with a training objective"
                )
        artifact_names = [artifact.name for artifact in self.artifacts]
        if len(set(artifact_names)) != len(artifact_names):
            raise ValueError(f"model {self.name!r} declares duplicate artifacts")
        if self.artifact_factory is not None and not callable(self.artifact_factory):
            raise TypeError(f"model {self.name!r} artifact_factory must be callable")
        if self.artifact_factory is not None and not self.artifacts:
            raise ValueError(
                f"model {self.name!r} declares artifact_factory without artifacts"
            )

    @property
    def output_type(self) -> Literal["point", "quantile", "distribution"]:
        """Declared public output kind, derived from orthogonal capabilities."""
        if "quantile-output" in self.capabilities:
            return "quantile"
        if "distribution-output" in self.capabilities:
            return "distribution"
        return "point"

    @property
    def task_modes(self) -> frozenset[str]:
        """Return supported public data settings from orthogonal capabilities."""
        mapping = {
            "time-series": "time_series",
            "spatiotemporal": "spatiotemporal",
            "covariate": "covariate",
        }
        return frozenset(mapping[key] for key in mapping if key in self.capabilities)

    def validate_params(self, params: dict) -> dict:
        unknown = sorted(set(params) - set(self.params_schema.model_fields))
        if unknown:
            raise ValueError(
                f"Unknown parameters for {self.name}: {', '.join(unknown)}"
            )
        return self.params_schema.model_validate(params).model_dump()

    def build(self, cfg, params: dict):
        return self.factory(cfg, self.validate_params(params))

    def build_with_artifacts(self, cfg, params: dict, paths: dict):
        """Construct an artifact-backed model through its explicit factory."""
        if self.artifact_factory is None:
            raise ValueError(
                f"model {self.name!r} declares runtime artifacts but has no artifact_factory"
            )
        return self.artifact_factory(cfg, self.validate_params(params), dict(paths))


class ModelCatalog:
    def __init__(self, refs: dict[str, str]) -> None:
        self._refs = dict(refs)
        self._loaded: dict[str, ModelSpec] = {}

    def names(self) -> list[str]:
        return sorted(self._refs)

    def refs(self) -> dict[str, str]:
        return dict(self._refs)

    def get(self, name: str) -> ModelSpec:
        if name in self._loaded:
            return self._loaded[name]
        module_path = self._refs.get(name)
        if module_path is None:
            available = ", ".join(self.names())
            raise KeyError(f"Unknown model {name!r}. Available: {available}")
        module = importlib.import_module(module_path)
        spec = getattr(module, "SPEC", None)
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"{module_path} must expose a ModelSpec named SPEC")
        if spec.name != name:
            raise ValueError(
                f"catalog key {name!r} disagrees with {module_path}.SPEC.name={spec.name!r}"
            )
        self._loaded[name] = spec
        return spec


MODEL_CATALOG = ModelCatalog({
    'ReFocus': 'tsflab.models.refocus.spec',
    'FreqMoE': 'tsflab.models.freqmoe.spec',
    'SWIFT': 'tsflab.models.swift.spec',
    'Sensorformer': 'tsflab.models.sensorformer.spec',
    'BiMamba': 'tsflab.models.bimamba.spec',
    'WPMixer': 'tsflab.models.wpmixer.spec',
    'DLinear': 'tsflab.models.dlinear.spec',
    'Linear': 'tsflab.models.linear.spec',
    'NLinear': 'tsflab.models.nlinear.spec',
    'RLinear': 'tsflab.models.rlinear.spec',
    'CMoS': 'tsflab.models.cmos.spec',
    'CycleNet': 'tsflab.models.cyclenet.spec',
    'TimeEmb': 'tsflab.models.timeemb.spec',
    'MixLinear': 'tsflab.models.mixlinear.spec',
    'PWS': 'tsflab.models.pws.spec',
    'PaiFilter': 'tsflab.models.paifilter.spec',
    'FITS': 'tsflab.models.fits.spec',
    'SVTime': 'tsflab.models.svtime.spec',
    'SparseTSF': 'tsflab.models.sparsetsf.spec',
    'TexFilter': 'tsflab.models.texfilter.spec',
    'Autoformer': 'tsflab.models.autoformer.spec',
    'FEDformer': 'tsflab.models.fedformer.spec',
    'PatchTST': 'tsflab.models.patchtst.spec',
    'PatchMLP': 'tsflab.models.patchmlp.spec',
    'xPatch': 'tsflab.models.xpatch.spec',
    'Amplifier': 'tsflab.models.amplifier.spec',
    'CrossLinear': 'tsflab.models.crosslinear.spec',
    'TimeBase': 'tsflab.models.timebase.spec',
    'TimeBridge': 'tsflab.models.timebridge.spec',
    'SegRNN': 'tsflab.models.segrnn.spec',
    'TSMixer': 'tsflab.models.tsmixer.spec',
    'LightTS': 'tsflab.models.lightts.spec',
    'SCINet': 'tsflab.models.scinet.spec',
    'TiDE': 'tsflab.models.tide.spec',
    'TimeMixer': 'tsflab.models.timemixer.spec',
    'TimesNet': 'tsflab.models.timesnet.spec',
    'iTransformer': 'tsflab.models.itransformer.spec',
    'STNorm': 'tsflab.models.stnorm.spec',
    'TimeXer': 'tsflab.models.timexer.spec',
    'TimeFilter': 'tsflab.models.timefilter.spec',
    'MambaSimple': 'tsflab.models.mambasimple.spec',
    'S_Mamba': 'tsflab.models.s_mamba.spec',
    'S4': 'tsflab.models.s4.spec',
    'MSGNet': 'tsflab.models.msgnet.spec',
    'HDMixer': 'tsflab.models.hdmixer.spec',
    'DSFormer': 'tsflab.models.dsformer.spec',
    'UMixer': 'tsflab.models.umixer.spec',
    'TimeKAN': 'tsflab.models.timekan.spec',
    'Fredformer': 'tsflab.models.fredformer.spec',
    'PAttn': 'tsflab.models.pattn.spec',
    'CARD': 'tsflab.models.card.spec',
    'NHiTS': 'tsflab.models.nhits.spec',
    'NBeats': 'tsflab.models.nbeats.spec',
    'DUET': 'tsflab.models.duet.spec',
    'ETSformer': 'tsflab.models.etsformer.spec',
    'NSTransformer': 'tsflab.models.nstransformer.spec',
    'SOFTS': 'tsflab.models.softs.spec',
    'Transformer': 'tsflab.models.transformer.spec',
    'Reformer': 'tsflab.models.reformer.spec',
    'Pyraformer': 'tsflab.models.pyraformer.spec',
    'MultiPatchFormer': 'tsflab.models.multipatchformer.spec',
    'ModernTCN': 'tsflab.models.moderntcn.spec',
    'Crossformer': 'tsflab.models.crossformer.spec',
    'FreTS': 'tsflab.models.frets.spec',
    'FiLM': 'tsflab.models.film.spec',
    'MICN': 'tsflab.models.micn.spec',
    'Koopa': 'tsflab.models.koopa.spec',
    'Informer': 'tsflab.models.informer.spec',
    'MTSMixer': 'tsflab.models.mtsmixer.spec',
    'Pathformer': 'tsflab.models.pathformer.spec',
    'WaveNet': 'tsflab.models.wavenet.spec',
    'DeepAR': 'tsflab.models.deepar.spec',
    'Sumba': 'tsflab.models.sumba.spec',
    'SRSNet': 'tsflab.models.srsnet.spec',
    'DTAF': 'tsflab.models.dtaf.spec',
    'TimePerceiver': 'tsflab.models.timeperceiver.spec',
    'CrossGNN': 'tsflab.models.crossgnn.spec',
    'RidgeRegressionTS': 'tsflab.models.ridge_regression_ts.spec',
    'LassoRegressionTS': 'tsflab.models.lasso_regression_ts.spec',
    'ElasticNetTS': 'tsflab.models.elastic_net_ts.spec',
    'BayesianRidgeTS': 'tsflab.models.bayesian_ridge_ts.spec',
    'PolynomialRegressionTS': 'tsflab.models.polynomial_regression_ts.spec',
    'KNNForecasterTS': 'tsflab.models.knn_forecaster_ts.spec',
    'SVRForecasterTS': 'tsflab.models.svr_forecaster_ts.spec',
    'GaussianProcessTS': 'tsflab.models.gaussian_process_ts.spec',
    'DecisionTreeTS': 'tsflab.models.decision_tree_ts.spec',
    'RandomForestTS': 'tsflab.models.random_forest_ts.spec',
    'ExtraTreesTS': 'tsflab.models.extra_trees_ts.spec',
    'GradientBoostingTS': 'tsflab.models.gradient_boosting_ts.spec',
    'XGBoostTS': 'tsflab.models.xgboost_ts.spec',
    'LightGBMTS': 'tsflab.models.lightgbm_ts.spec',
    'CatBoostTS': 'tsflab.models.catboost_ts.spec',
    'ARIMATS': 'tsflab.models.arima_ts.spec',
    'AutoRegressiveTS': 'tsflab.models.autoregressive_ts.spec',
    'ExpSmoothingTS': 'tsflab.models.exp_smoothing_ts.spec',
    'KalmanFilterTS': 'tsflab.models.kalman_filter_ts.spec',
    'MLPForecasterTS': 'tsflab.models.mlp_forecaster_ts.spec',
    'RNNForecasterTS': 'tsflab.models.rnn_forecaster_ts.spec',
    'GRUForecasterTS': 'tsflab.models.gru_forecaster_ts.spec',
    'LSTMForecasterTS': 'tsflab.models.lstm_forecaster_ts.spec',
    'TCNForecasterTS': 'tsflab.models.tcn_forecaster_ts.spec',
    'Aurora': 'tsflab.models.aurora.spec',
    'CRIB': 'tsflab.models.crib.spec',
    'TimeAlign': 'tsflab.models.timealign.spec',
    'GTR': 'tsflab.models.gtr.spec',
    'PhaseFormer': 'tsflab.models.phaseformer.spec',
    'PMDformer': 'tsflab.models.pmdformer.spec',
    'MMPD': 'tsflab.models.mmpd.spec',
    'COSA': 'tsflab.models.cosa.spec',
    'Composed': 'tsflab.models.composed.spec',
    'DistDF': 'tsflab.models.distdf.spec',
    'Sonnet': 'tsflab.models.sonnet.spec',
    'APN': 'tsflab.models.apn.spec',
    'TimeCAP': 'tsflab.models.timecap.spec',
    'GOTSF': 'tsflab.models.gotsf.spec',
    'FTP': 'tsflab.models.ftp.spec',
    'OccamVTS': 'tsflab.models.occamvts.spec',
    'HN_MVTS': 'tsflab.models.hn_mvts.spec',
    'SEMPO': 'tsflab.models.sempo.spec',
    'InterPDN': 'tsflab.models.interpdn.spec',
    'TimeO1': 'tsflab.models.timeo1.spec',
    'FeTS': 'tsflab.models.fets.spec',
    'SymTime': 'tsflab.models.symtime.spec',
    'ImplicitForecaster': 'tsflab.models.implicitforecaster.spec',
    'AMRC': 'tsflab.models.amrc.spec',
    'HMformer': 'tsflab.models.hmformer.spec',
    'TiRex': 'tsflab.models.tirex.spec',
    'GlocalIB': 'tsflab.models.glocalib.spec',
    'QuantileDLinear': 'tsflab.models.quantile_dlinear.spec',
    'QuantilePatchTST': 'tsflab.models.quantile_patchtst.spec',
    'MQRNN': 'tsflab.models.mqrnn.spec',
    'GaussianMLP': 'tsflab.models.gaussian_mlp.spec',
    'LatentTSF': 'tsflab.models.latenttsf.spec',
    'CoRA': 'tsflab.models.cora.spec',
    'DynamicTMoE': 'tsflab.models.dynamic_tmoe.spec',
    'PULSE': 'tsflab.models.pulse.spec',
    'OLinear': 'tsflab.models.olinear.spec',
    'MAFS': 'tsflab.models.mafs.spec',
    'TSRAG': 'tsflab.models.tsrag.spec',
    'TimeMosaic': 'tsflab.models.timemosaic.spec',
    'Kronos': 'tsflab.models.kronos.spec',
    'MoFo': 'tsflab.models.mofo.spec',
    'PHAT': 'tsflab.models.phat.spec',
    'BiST': 'tsflab.models.bist.spec',
    'MAGE': 'tsflab.models.mage.spec',
    'STOP': 'tsflab.models.stop.spec',
    'CauAir': 'tsflab.models.cauair.spec',
    'AirCade': 'tsflab.models.aircade.spec',
    'GTS': 'tsflab.models.gts.spec',
    'STID': 'tsflab.models.stid.spec',
    'GWNet': 'tsflab.models.gwnet.spec',
    'D2STGNN': 'tsflab.models.d2stgnn.spec',
    'DFDGCN': 'tsflab.models.dfdgcn.spec',
    'STGCN': 'tsflab.models.stgcn.spec',
    'AGCRN': 'tsflab.models.agcrn.spec',
    'DCRNN': 'tsflab.models.dcrnn.spec',
    'StemGNN': 'tsflab.models.stemgnn.spec',
    'MTGNN': 'tsflab.models.mtgnn.spec',
    'STGODE': 'tsflab.models.stgode.spec',
    'STAEformer': 'tsflab.models.staeformer.spec',
    'DGCRN': 'tsflab.models.dgcrn.spec',
    'STDN': 'tsflab.models.stdn.spec',
    'STPGNN': 'tsflab.models.stpgnn.spec',
    'MegaCRN': 'tsflab.models.megacrn.spec',
    'HimNet': 'tsflab.models.himnet.spec',
    'STWave': 'tsflab.models.stwave.spec',
    'BigST': 'tsflab.models.bigst.spec',
    'ASTGCN': 'tsflab.models.astgcn.spec',
    'GCLSTM': 'tsflab.models.gclstm.spec',
    'DeepAir': 'tsflab.models.deepair.spec',
    'STTN': 'tsflab.models.sttn.spec',
    'GAGNN': 'tsflab.models.gagnn.spec',
    'PM25_GNN': 'tsflab.models.pm25gnn.spec',
    'AirFormer': 'tsflab.models.airformer.spec',
    'DSTAGNN': 'tsflab.models.dstagnn.spec',
    'PCDCNet': 'tsflab.models.pcdcnet.spec',
    'AirPhyNet': 'tsflab.models.airphynet.spec',
    'AirDualODE': 'tsflab.models.airdualode.spec',
    'HL': 'tsflab.models.hl.spec',
    'LSTM': 'tsflab.models.lstm.spec',
    'RPMixer': 'tsflab.models.rpmixer.spec',
    'MGSFformer': 'tsflab.models.mgsfformer.spec',
    'CATS': 'tsflab.models.cats.spec',
    'DPWMixer': 'tsflab.models.dpwmixer.spec',
    'AWEMixer': 'tsflab.models.awemixer.spec',
    'TimeExpert': 'tsflab.models.timeexpert.spec',
    'WDformer': 'tsflab.models.wdformer.spec',
    'LSINet': 'tsflab.models.lsinet.spec',
    'Dualformer': 'tsflab.models.dualformer.spec',
    'SEMixer': 'tsflab.models.semixer.spec',
    'SDMixer': 'tsflab.models.sdmixer.spec',
    'VisiFold': 'tsflab.models.visifold.spec',
    'Extralonger': 'tsflab.models.extralonger.spec',
    'ST-SSDL': 'tsflab.models.st_ssdl.spec',
    'RAGC': 'tsflab.models.ragc.spec',
    "TQNet": "tsflab.models.tqnet.spec",
    "Gateformer": "tsflab.models.gateformer.spec",
    "TimePro": "tsflab.models.timepro.spec",
    "CANet": "tsflab.models.canet.spec",
    "CoRe": "tsflab.models.core.spec",
    "SAMformer": "tsflab.models.samformer.spec",
    "AMD": "tsflab.models.amd.spec",
    "MoU": "tsflab.models.mou.spec",
    "PatchTSMixer": "tsflab.models.patchtsmixer.spec",
    "AdaMSHyper": "tsflab.models.adamshyper.spec",
    "PENGUIN": "tsflab.models.penguin.spec",
    "DiPELinear": "tsflab.models.dipelinear.spec",
    "PGN": "tsflab.models.pgn.spec",
    "FreDF": "tsflab.models.fredf.spec",
    "GPHT": "tsflab.models.gpht.spec",
    "SAMBA": "tsflab.models.samba.spec",
    "MTLinear": "tsflab.models.mtlinear.spec",
    "LIFT": "tsflab.models.lift.spec",
    "TimeMachine": "tsflab.models.timemachine.spec",
    "MambaTS": "tsflab.models.mambats.spec",
    "ARMD": "tsflab.models.armd.spec",
    "STDMAE": "tsflab.models.stdmae.spec",
    "CALF": "tsflab.models.calf.spec",
    "Basisformer": "tsflab.models.basisformer.spec",
    "EMAformer": "tsflab.models.emaformer.spec",
    "D3VAE": "tsflab.models.d3vae.spec",
    "Client": "tsflab.models.client.spec",
    "GWNetGMM": "tsflab.models.gwnet_gmm.spec",
    "NsDiff": "tsflab.models.nsdiff.spec",
    "FSatten": "tsflab.models.fsatten.spec",
    "OCETS": "tsflab.models.ocets.spec",
    "EvoMSN": "tsflab.models.evomsn.spec",
    "CLoRA": "tsflab.models.clora.spec",
    "DropoutTS": "tsflab.models.dropoutts.spec",
    "OLMA": "tsflab.models.olma.spec",
    "AutoTimes": "tsflab.models.autotimes.spec",
    "LLM4TS": "tsflab.models.llm4ts.spec",
    "CosDir": "tsflab.models.cosdir.spec",
    "DB2TransF": "tsflab.models.db2_transf.spec",
    "PCATransformer": "tsflab.models.pcatransformer.spec",
    "RILoss": "tsflab.models.riloss.spec",
    "WinNet": "tsflab.models.winnet.spec",
    "DecompRevisit": "tsflab.models.decomprevisit.spec",
    "SST": "tsflab.models.sst.spec",
    "RootPurge": "tsflab.models.rootpurge.spec",
    "APTF": "tsflab.models.aptf.spec",
    "ARKAN": "tsflab.models.arkan.spec",
    "WDAN": "tsflab.models.wdan.spec",
    "AROpt": "tsflab.models.aropt.spec",
    "STLoRA": "tsflab.models.stlora.spec",
    "STELLA": "tsflab.models.stella.spec",
    "LMSAutoTSF": "tsflab.models.lmsautotsf.spec",
    "LiNo": "tsflab.models.lino.spec",
    "STHD": "tsflab.models.sthd.spec",
    "FAITH": "tsflab.models.faith.spec",
    "PRformer": "tsflab.models.prformer.spec",
    "WFTNet": "tsflab.models.wftnet.spec",
    "GCformer": "tsflab.models.gcformer.spec",
    "GBT": "tsflab.models.gbt.spec",
    "Times2D": "tsflab.models.times2d.spec",
    "TSLIF": "tsflab.models.tslif.spec",
    "FDNet": "tsflab.models.fdnet.spec",
    "FPPformer": "tsflab.models.fppformer.spec",
    "ReNF": "tsflab.models.renf.spec",
    "VCformer": "tsflab.models.vcformer.spec",
    "MambaProbTSF": "tsflab.models.mambaprobtsf.spec",
    "KARMA": "tsflab.models.karma.spec",
    "Minusformer": "tsflab.models.minusformer.spec",
    "TFPS": "tsflab.models.tfps.spec",
    "SSCLForecaster": "tsflab.models.ssclforecaster.spec",
    "Esiformer": "tsflab.models.esiformer.spec",
    "ASGMamba": "tsflab.models.asgmamba.spec",
    "SCAM": "tsflab.models.scam.spec",
    "AdaMamba": "tsflab.models.adamamba.spec",
    "GraphLassoMTSF": "tsflab.models.graphlassomtsf.spec",
    "TSRM": "tsflab.models.tsrm.spec",
    "DRFormer": "tsflab.models.drformer.spec",
    "TFKAN": "tsflab.models.tfkan.spec",
    "PRISM": "tsflab.models.prism.spec",
    "WaveRoRA": "tsflab.models.waverora.spec",
    "GSWaN": "tsflab.models.gswan.spec",
    "DMSC": "tsflab.models.dmsc.spec",
    "SCFormer": "tsflab.models.scformer.spec",
    "SARAF": "tsflab.models.saraf.spec",
    "CDPM": "tsflab.models.cdpm.spec",
    "MultiSPANS": "tsflab.models.multispans.spec",
    "TimeGS": "tsflab.models.timegs.spec",
    "PPGF": "tsflab.models.ppgf.spec",
    "PPM": "tsflab.models.ppm.spec",
    "TwinFormer": "tsflab.models.twinformer.spec",
    "HADL": "tsflab.models.hadl.spec",
    "FRWKV": "tsflab.models.frwkv.spec",
    "FRWKVPlus": "tsflab.models.frwkvplus.spec",
    "Ister": "tsflab.models.ister.spec",
    "GAMMANet": "tsflab.models.gammanet.spec",
    "Dozerformer": "tsflab.models.dozerformer.spec",
    "REPNet": "tsflab.models.repnet.spec",
    "SDGFNet": "tsflab.models.sdgfnet.spec",
    "DPANet": "tsflab.models.dpanet.spec",
    "VLBM": "tsflab.models.vlbm.spec",
    "DRAGON": "tsflab.models.dragon.spec",
    "CAPS": "tsflab.models.caps.spec",
    "MoGU": "tsflab.models.mogu.spec",
    "ReCycle": "tsflab.models.recycle.spec",
    "SiamTST": "tsflab.models.siamtst.spec",
    "GMRL": "tsflab.models.gmrl.spec",
    "MoSSL": "tsflab.models.mossl.spec",
    "VolDyVAE": "tsflab.models.voldyvae.spec",
    "GRAMODE": "tsflab.models.gramode.spec",
    "SORMamba": "tsflab.models.sormamba.spec",
    "MFRS": "tsflab.models.mfrs.spec",
    "MTSUNMixers": "tsflab.models.mtsunmixers.spec",
    "CCDM": "tsflab.models.ccdm.spec",
    "TALON": "tsflab.models.talon.spec",
    "DynGDiff": "tsflab.models.dyngdiff.spec",
    "StaTS": "tsflab.models.stats.spec",
    "LogTrans": "tsflab.models.logtrans.spec",
    "TimeGrad": "tsflab.models.timegrad.spec",
})
