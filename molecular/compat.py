# compat.py
# PyTorch Lightning / torchmetrics compatibility layer.
# Provides fallback implementations so the pipeline runs with plain PyTorch
# when pytorch_lightning and/or torchmetrics are not installed.

import os
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detect available packages
# ---------------------------------------------------------------------------
try:
    import pytorch_lightning as pl
    HAS_LIGHTNING = True
    LIGHTNING_VERSION = pl.__version__
except ImportError:
    HAS_LIGHTNING = False
    LIGHTNING_VERSION = "not installed"

try:
    import torchmetrics  # noqa: F401
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

_logger.info(f"PyTorch Lightning: {LIGHTNING_VERSION}")
_logger.info(f"torchmetrics available: {HAS_TORCHMETRICS}")

# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------

def seed_everything(seed, workers=True):
    """Seed all RNGs for reproducibility."""
    if HAS_LIGHTNING:
        pl.seed_everything(seed, workers=workers)
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if workers:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ===================================================================
# BASE CLASSES
# ===================================================================

if HAS_LIGHTNING:
    LightningModuleBase = pl.LightningModule
    LightningDataModuleBase = pl.LightningDataModule
    CallbackBase = pl.Callback
else:
    # ------------------------------------------------------------------
    # LightningModule fallback
    # ------------------------------------------------------------------
    class LightningModuleBase(nn.Module):
        """Minimal pl.LightningModule replacement using plain PyTorch."""

        def __init__(self):
            super().__init__()
            self._hparams = {}
            self.trainer = None
            # Epoch-level metric accumulator: name -> [(value, batch_size)]
            self._epoch_metrics = {}

        # --- hparams ---------------------------------------------------
        def save_hyperparameters(self, *args, **kwargs):
            import inspect
            frame = inspect.currentframe().f_back
            local_vars = frame.f_locals
            if args:
                for name in args:
                    if name in local_vars:
                        self._hparams[name] = local_vars[name]
            # ignore_keys for compat
            ignore = kwargs.get('ignore', [])
            for name in ignore:
                self._hparams.pop(name, None)

        @property
        def hparams(self):
            return self._hparams

        # --- logging ---------------------------------------------------
        def log(self, name, value, **kwargs):
            pass  # no-op; SimpleTrainer reads metrics from epoch accumulator

        def log_dict(self, metrics, on_step=False, on_epoch=True,
                     prog_bar=False, logger=True, batch_size=None):
            bs = batch_size or 1
            for name, value in metrics.items():
                val = value.detach().item() if isinstance(value, torch.Tensor) else float(value)
                if on_epoch:
                    self._epoch_metrics.setdefault(name, []).append((val, bs))

        # --- internal helpers used by SimpleTrainer --------------------
        def _compute_epoch_metrics(self):
            """Weighted average of accumulated epoch metrics."""
            result = {}
            for name, entries in self._epoch_metrics.items():
                total_w = sum(w for _, w in entries)
                wsum = sum(v * w for v, w in entries)
                result[name] = wsum / total_w if total_w > 0 else 0.0
            return result

        def _reset_epoch_metrics(self):
            self._epoch_metrics = {}

        # --- hooks (no-ops unless overridden) --------------------------
        def on_train_start(self):
            pass

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, **kwargs):
            raise NotImplementedError(
                "load_from_checkpoint is not available without PyTorch Lightning. "
                "Use torch.load() + model.load_state_dict() instead."
            )

    # ------------------------------------------------------------------
    # LightningDataModule fallback
    # ------------------------------------------------------------------
    class LightningDataModuleBase:
        """Minimal pl.LightningDataModule replacement."""

        def __init__(self):
            pass

        def setup(self, stage=None):
            pass

        def teardown(self, stage=None):
            pass

        def train_dataloader(self):
            raise NotImplementedError

        def val_dataloader(self):
            raise NotImplementedError

        def test_dataloader(self):
            raise NotImplementedError

    # ------------------------------------------------------------------
    # Callback fallback
    # ------------------------------------------------------------------
    class CallbackBase:
        """Minimal pl.Callback replacement."""

        def on_train_epoch_end(self, trainer, pl_module):
            pass

        def on_fit_start(self, trainer, pl_module):
            pass

        def on_fit_end(self, trainer, pl_module):
            pass


# ===================================================================
# CALLBACKS
# ===================================================================

if HAS_LIGHTNING:
    from pytorch_lightning.callbacks import (
        EarlyStopping as EarlyStopping,
        ModelCheckpoint as ModelCheckpoint,
    )
else:
    class EarlyStopping:
        """Plain-PyTorch early-stopping callback."""

        def __init__(self, monitor='val_loss', mode='min', patience=10,
                     verbose=True, **kwargs):
            self.monitor = monitor
            self.mode = mode
            self.patience = patience
            self.verbose = verbose
            self.best_score = None
            self.counter = 0
            self.should_stop = False

        def check(self, metrics):
            score = metrics.get(self.monitor)
            if score is None:
                return
            if self.best_score is None:
                self.best_score = score
            elif self._is_better(score):
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    _logger.info(
                        f"EarlyStopping: {self.counter}/{self.patience} "
                        f"(best {self.monitor}: {self.best_score:.6f})"
                    )
                if self.counter >= self.patience:
                    self.should_stop = True
                    if self.verbose:
                        _logger.info("EarlyStopping triggered")

        def _is_better(self, score):
            if self.mode == 'min':
                return score < self.best_score
            return score > self.best_score

    class ModelCheckpoint:
        """Plain-PyTorch model checkpointing callback."""

        def __init__(self, dirpath='', filename='checkpoint', save_top_k=1,
                     monitor='val_loss', mode='min', save_last=False, **kwargs):
            self.dirpath = dirpath
            self.filename = filename
            self.save_top_k = save_top_k
            self.monitor = monitor
            self.mode = mode
            self.save_last = save_last
            self.best_score = None
            self.best_model_path = ""
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)

        def check(self, metrics, model):
            score = metrics.get(self.monitor)
            if score is None:
                return
            if self.best_score is None or self._is_better(score):
                self.best_score = score
                path = os.path.join(self.dirpath, f"{self.filename}.ckpt")
                torch.save(model.state_dict(), path)
                self.best_model_path = path

        def _is_better(self, score):
            if self.mode == 'min':
                return score < self.best_score
            return score > self.best_score


# ===================================================================
# LOGGER
# ===================================================================

if HAS_LIGHTNING:
    TensorBoardLogger = pl.loggers.TensorBoardLogger
else:
    class TensorBoardLogger:
        """No-op TensorBoard logger when PL is absent."""

        def __init__(self, save_dir='', name='', version=None,
                     default_hp_metric=True, **kwargs):
            self.save_dir = save_dir
            self.name = name
            self.version = version

        def log_metrics(self, metrics, step=None):
            pass

        def log_hyperparams(self, params):
            pass


# ===================================================================
# TRAINER
# ===================================================================

if HAS_LIGHTNING:
    Trainer = pl.Trainer
else:
    class _TrainerState:
        def __init__(self):
            self.stage = 'train'

    class Trainer:
        """Minimal Trainer that runs standard PyTorch training loops."""

        def __init__(self, max_epochs=100, accelerator='auto', devices=1,
                     callbacks=None, logger=None, precision='32',
                     enable_progress_bar=True, deterministic=False,
                     enable_checkpointing=True, num_sanity_val_steps=0,
                     log_every_n_steps=50, benchmark=False, **kwargs):
            self.max_epochs = max_epochs
            self.callbacks = callbacks or []
            self._tb_logger = logger
            self.enable_progress_bar = enable_progress_bar
            self.callback_metrics = {}
            self.current_epoch = 0
            self.state = _TrainerState()
            self.datamodule = None
            self.final_training = False

            # Resolve device
            if accelerator == 'gpu' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif accelerator == 'auto':
                self.device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'
                )
            else:
                self.device = torch.device('cpu')

            # AMP
            self._use_amp = str(precision) in ('16', '16-mixed')

        # ---------------------------------------------------------------
        def fit(self, model, datamodule=None, train_dataloaders=None,
                val_dataloaders=None):
            """Run the full training loop."""
            self.datamodule = datamodule
            model.trainer = self
            model = model.to(self.device)

            # Dataloaders
            if datamodule is not None:
                if hasattr(datamodule, 'setup'):
                    datamodule.setup('fit')
                train_loader = datamodule.train_dataloader()
                val_loader = datamodule.val_dataloader()
            else:
                train_loader = train_dataloaders
                val_loader = val_dataloaders

            # Optimiser / scheduler
            opt_cfg = model.configure_optimizers()
            if isinstance(opt_cfg, dict):
                optimizer = opt_cfg['optimizer']
                sched_cfg = opt_cfg.get('lr_scheduler')
                if isinstance(sched_cfg, dict):
                    scheduler = sched_cfg['scheduler']
                    sched_monitor = sched_cfg.get('monitor', 'val_loss')
                else:
                    scheduler, sched_monitor = None, None
            else:
                optimizer = opt_cfg
                scheduler, sched_monitor = None, None

            scaler = (
                torch.amp.GradScaler('cuda')
                if self._use_amp and self.device.type == 'cuda'
                else None
            )

            # Callback helpers
            early_stoppers = [
                cb for cb in self.callbacks if isinstance(cb, EarlyStopping)
            ]
            checkpointers = [
                cb for cb in self.callbacks if isinstance(cb, ModelCheckpoint)
            ]

            # Notify fit start
            for cb in self.callbacks:
                if hasattr(cb, 'on_fit_start'):
                    cb.on_fit_start(self, model)
            if hasattr(model, 'on_train_start'):
                model.on_train_start()

            # ---- Epoch loop -------------------------------------------
            for epoch in range(self.max_epochs):
                self.current_epoch = epoch
                self.state.stage = 'train'
                model.train()

                if hasattr(model, '_reset_epoch_metrics'):
                    model._reset_epoch_metrics()
                # Reset fallback metric accumulators on model
                self._reset_model_metrics(model)

                for batch_idx, batch in enumerate(train_loader):
                    batch = self._to_device(batch)
                    optimizer.zero_grad()
                    if self._use_amp and scaler is not None:
                        with torch.amp.autocast('cuda'):
                            loss = model.training_step(batch, batch_idx)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss = model.training_step(batch, batch_idx)
                        loss.backward()
                        optimizer.step()

                train_metrics = {}
                if hasattr(model, '_compute_epoch_metrics'):
                    train_metrics = model._compute_epoch_metrics()

                # Callback: on_train_epoch_end
                for cb in self.callbacks:
                    if hasattr(cb, 'on_train_epoch_end'):
                        cb.on_train_epoch_end(self, model)

                # ---- Validation ---------------------------------------
                if val_loader is not None:
                    self.state.stage = 'validate'
                    model.eval()
                    if hasattr(model, '_reset_epoch_metrics'):
                        model._reset_epoch_metrics()
                    self._reset_model_metrics(model)

                    with torch.no_grad():
                        for batch_idx, batch in enumerate(val_loader):
                            batch = self._to_device(batch)
                            model.validation_step(batch, batch_idx)

                    val_metrics = {}
                    if hasattr(model, '_compute_epoch_metrics'):
                        val_metrics = model._compute_epoch_metrics()

                    # Override with proper epoch-level metric values
                    val_metrics = self._override_with_metric_computes(
                        model, val_metrics, prefix='val_'
                    )
                    train_metrics = self._override_with_metric_computes(
                        model, train_metrics, prefix='train_'
                    )

                    all_metrics = {**train_metrics, **val_metrics}
                    self.callback_metrics = {
                        k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
                        for k, v in all_metrics.items()
                    }

                    if self.enable_progress_bar:
                        parts = [
                            f"{k}: {v:.4f}" for k, v in all_metrics.items()
                            if isinstance(v, (int, float))
                        ]
                        _logger.info(f"Epoch {epoch}: {', '.join(parts)}")

                    # Callbacks: early stopping & checkpointing
                    for es in early_stoppers:
                        es.check(all_metrics)
                    for cp in checkpointers:
                        cp.check(all_metrics, model)

                    # Scheduler step
                    if scheduler is not None:
                        mon = (
                            all_metrics.get(sched_monitor)
                            if sched_monitor else None
                        )
                        if mon is not None:
                            scheduler.step(mon)

                    if any(es.should_stop for es in early_stoppers):
                        _logger.info(f"Early stopping at epoch {epoch}")
                        break

            # Notify fit end
            for cb in self.callbacks:
                if hasattr(cb, 'on_fit_end'):
                    cb.on_fit_end(self, model)

        # ---------------------------------------------------------------
        # helpers
        # ---------------------------------------------------------------
        def _to_device(self, batch):
            if isinstance(batch, (tuple, list)):
                return tuple(
                    item.to(self.device) if hasattr(item, 'to') else item
                    for item in batch
                )
            if hasattr(batch, 'to'):
                return batch.to(self.device)
            return batch

        @staticmethod
        def _reset_model_metrics(model):
            """Reset all FallbackMetric instances on the model."""
            for attr_name in dir(model):
                attr = getattr(model, attr_name, None)
                if isinstance(attr, _FallbackMetric):
                    attr.reset()

        @staticmethod
        def _override_with_metric_computes(model, metrics_dict, prefix):
            """Replace averaged log_dict values with proper epoch-level
            metric computes from FallbackMetric objects."""
            # Build map: metric_key -> attribute name
            # e.g. 'val_auc' -> 'val_auc' attribute on model
            for key in list(metrics_dict.keys()):
                if not key.startswith(prefix):
                    continue
                attr = getattr(model, key, None)
                if isinstance(attr, _FallbackMetric):
                    computed = attr.compute()
                    if computed is not None:
                        metrics_dict[key] = (
                            computed.item()
                            if isinstance(computed, torch.Tensor)
                            else float(computed)
                        )
            return metrics_dict


# ===================================================================
# FALLBACK METRIC CLASSES (used when torchmetrics is not installed)
# ===================================================================

class _FallbackMetric:
    """Base for sklearn-backed metric accumulators.

    Accumulates predictions and targets across batches and computes
    the epoch-level metric via sklearn at the end.
    """

    def __init__(self):
        self._preds = []
        self._targets = []

    def __call__(self, preds, targets):
        self._preds.append(preds.detach().cpu())
        self._targets.append(targets.detach().cpu())
        return torch.tensor(0.0)  # placeholder batch value

    def compute(self):
        if not self._preds:
            return torch.tensor(0.0)
        all_preds = torch.cat(self._preds)
        all_targets = torch.cat(self._targets)
        try:
            return torch.tensor(self._compute(all_preds, all_targets))
        except Exception:
            return torch.tensor(0.0)

    def _compute(self, preds, targets):
        raise NotImplementedError

    def reset(self):
        self._preds = []
        self._targets = []

    def to(self, device):
        return self


# ---- Multiclass metrics ------------------------------------------------

class _MulticlassAUROC(_FallbackMetric):
    def __init__(self, num_classes, **kw):
        super().__init__()
        self.num_classes = num_classes

    def _compute(self, preds, targets):
        return roc_auc_score(
            targets.numpy(), preds.numpy(),
            multi_class='ovr', average='macro'
        )


class _MulticlassAccuracy(_FallbackMetric):
    def __init__(self, num_classes, **kw):
        super().__init__()
        self.num_classes = num_classes

    def _compute(self, preds, targets):
        return accuracy_score(targets.numpy(), preds.numpy().argmax(axis=1))


class _MulticlassF1Score(_FallbackMetric):
    def __init__(self, num_classes, average='macro', **kw):
        super().__init__()
        self.average = average

    def _compute(self, preds, targets):
        return f1_score(
            targets.numpy(), preds.numpy().argmax(axis=1),
            average=self.average, zero_division=0
        )


class _MulticlassPrecision(_FallbackMetric):
    def __init__(self, num_classes, average='macro', **kw):
        super().__init__()
        self.average = average

    def _compute(self, preds, targets):
        return precision_score(
            targets.numpy(), preds.numpy().argmax(axis=1),
            average=self.average, zero_division=0
        )


class _MulticlassRecall(_FallbackMetric):
    def __init__(self, num_classes, average='macro', **kw):
        super().__init__()
        self.average = average

    def _compute(self, preds, targets):
        return recall_score(
            targets.numpy(), preds.numpy().argmax(axis=1),
            average=self.average, zero_division=0
        )


class _MulticlassMatthewsCorrCoef(_FallbackMetric):
    def __init__(self, num_classes, **kw):
        super().__init__()

    def _compute(self, preds, targets):
        return matthews_corrcoef(targets.numpy(), preds.numpy().argmax(axis=1))


class _MulticlassCohenKappa(_FallbackMetric):
    def __init__(self, num_classes, **kw):
        super().__init__()

    def _compute(self, preds, targets):
        return cohen_kappa_score(targets.numpy(), preds.numpy().argmax(axis=1))


class _MulticlassAveragePrecision(_FallbackMetric):
    def __init__(self, num_classes, **kw):
        super().__init__()
        self.num_classes = num_classes

    def _compute(self, preds, targets):
        return average_precision_score(
            targets.numpy(), preds.numpy(), average='macro'
        )


class _MulticlassSpecificity(_FallbackMetric):
    """Macro-averaged specificity (TNR) across classes."""
    def __init__(self, num_classes, average='macro', **kw):
        super().__init__()
        self.num_classes = num_classes

    def _compute(self, preds, targets):
        y_true = targets.numpy()
        y_pred = preds.numpy().argmax(axis=1)
        specs = []
        for c in range(self.num_classes):
            tn = ((y_pred != c) & (y_true != c)).sum()
            fp = ((y_pred == c) & (y_true != c)).sum()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specs.append(spec)
        return float(np.mean(specs))


# ---- Binary metrics -----------------------------------------------------

class _BinaryAUROC(_FallbackMetric):
    def __init__(self, **kw):
        super().__init__()

    def _compute(self, preds, targets):
        return roc_auc_score(targets.numpy(), preds.numpy())


class _BinaryAccuracy(_FallbackMetric):
    def _compute(self, preds, targets):
        return accuracy_score(targets.numpy(), (preds.numpy() > 0.5).astype(int))


class _BinaryF1Score(_FallbackMetric):
    def _compute(self, preds, targets):
        return f1_score(targets.numpy(), (preds.numpy() > 0.5).astype(int), zero_division=0)


class _BinaryPrecision(_FallbackMetric):
    def _compute(self, preds, targets):
        return precision_score(targets.numpy(), (preds.numpy() > 0.5).astype(int), zero_division=0)


class _BinaryRecall(_FallbackMetric):
    def _compute(self, preds, targets):
        return recall_score(targets.numpy(), (preds.numpy() > 0.5).astype(int), zero_division=0)


class _BinarySpecificity(_FallbackMetric):
    def _compute(self, preds, targets):
        return recall_score(targets.numpy(), (preds.numpy() > 0.5).astype(int), pos_label=0, zero_division=0)


class _BinaryMatthewsCorrCoef(_FallbackMetric):
    def _compute(self, preds, targets):
        return matthews_corrcoef(targets.numpy(), (preds.numpy() > 0.5).astype(int))


class _BinaryAveragePrecision(_FallbackMetric):
    def _compute(self, preds, targets):
        return average_precision_score(targets.numpy(), preds.numpy())


class _BinaryCohenKappa(_FallbackMetric):
    def _compute(self, preds, targets):
        return cohen_kappa_score(targets.numpy(), (preds.numpy() > 0.5).astype(int))


# ---- AUROC / AveragePrecision (task-dispatched) -------------------------

class _AUROC_Dispatch(_FallbackMetric):
    """Mimics torchmetrics.AUROC(task=...)."""
    def __init__(self, task='binary', num_classes=None, **kw):
        super().__init__()
        self.task = task
        self.num_classes = num_classes

    def _compute(self, preds, targets):
        if self.task == 'binary':
            return roc_auc_score(targets.numpy(), preds.numpy())
        return roc_auc_score(
            targets.numpy(), preds.numpy(),
            multi_class='ovr', average='macro'
        )


class _AP_Dispatch(_FallbackMetric):
    """Mimics torchmetrics.AveragePrecision(task=...)."""
    def __init__(self, task='binary', num_classes=None, **kw):
        super().__init__()
        self.task = task

    def _compute(self, preds, targets):
        if self.task == 'binary':
            return average_precision_score(targets.numpy(), preds.numpy())
        return average_precision_score(
            targets.numpy(), preds.numpy(), average='macro'
        )


# ---- Regression metrics --------------------------------------------------

class _MAE(_FallbackMetric):
    def _compute(self, preds, targets):
        return mean_absolute_error(targets.numpy(), preds.numpy())


class _MSE(_FallbackMetric):
    def _compute(self, preds, targets):
        return mean_squared_error(targets.numpy(), preds.numpy())


class _R2(_FallbackMetric):
    def _compute(self, preds, targets):
        return r2_score(targets.numpy(), preds.numpy())


# ===================================================================
# PUBLIC METRIC FACTORIES  (used by model.py)
# ===================================================================
# When torchmetrics IS available we re-export the real classes.
# When it is NOT we return our sklearn fallbacks.

if HAS_TORCHMETRICS:
    from torchmetrics import AUROC, AveragePrecision
    from torchmetrics.classification import (
        BinaryAccuracy, BinaryF1Score, BinaryPrecision,
        BinaryRecall, BinarySpecificity, BinaryMatthewsCorrCoef,
        BinaryAUROC, BinaryAveragePrecision, BinaryCohenKappa,
        MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
        MulticlassRecall, MulticlassMatthewsCorrCoef,
        MulticlassAUROC, MulticlassAveragePrecision,
        MulticlassCohenKappa, MulticlassSpecificity,
    )
    from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
else:
    # Fallback aliases
    AUROC = _AUROC_Dispatch
    AveragePrecision = _AP_Dispatch

    BinaryAccuracy = _BinaryAccuracy
    BinaryF1Score = _BinaryF1Score
    BinaryPrecision = _BinaryPrecision
    BinaryRecall = _BinaryRecall
    BinarySpecificity = _BinarySpecificity
    BinaryMatthewsCorrCoef = _BinaryMatthewsCorrCoef
    BinaryAUROC = _BinaryAUROC
    BinaryAveragePrecision = _BinaryAveragePrecision
    BinaryCohenKappa = _BinaryCohenKappa

    MulticlassAccuracy = _MulticlassAccuracy
    MulticlassF1Score = _MulticlassF1Score
    MulticlassPrecision = _MulticlassPrecision
    MulticlassRecall = _MulticlassRecall
    MulticlassMatthewsCorrCoef = _MulticlassMatthewsCorrCoef
    MulticlassAUROC = _MulticlassAUROC
    MulticlassAveragePrecision = _MulticlassAveragePrecision
    MulticlassCohenKappa = _MulticlassCohenKappa
    MulticlassSpecificity = _MulticlassSpecificity

    MeanAbsoluteError = _MAE
    MeanSquaredError = _MSE
    R2Score = _R2
