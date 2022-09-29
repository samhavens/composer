# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Optional, Sequence, Type, Union

import torch
from torch.nn.modules import LayerNorm
from torch.optim import Optimizer

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)


class _RMSNorm(torch.nn.Module):
    """`Root Mean Square LayerNorm <https://arxiv.org/abs/1910.07467>`_ layer.

    LayerNorm without re-centering, so computationally more efficient

    Identical to LayerNorm when mean is 0

    Args:
        eps (float, optional): numerical stability constant. Default: ``1e-8``.

    Raises:
        Probably
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

    @classmethod
    def from_layer_norm(cls, module: torch.nn.Module, eps: float = 1e-8) -> "_RMSNorm":
        assert isinstance(module, LayerNorm), 'Module is not LayerNorm!'
        dim = len(module.normalized_shape)
        return cls(dim, eps)



def apply_rms_norm(
        model: torch.nn.Module,
        eps: float = 1e-8,
        optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None,
    ) -> torch.nn.Module:
    """`Root Mean Square LayerNorm <https://arxiv.org/abs/1910.07467>`_ layer.

    LayerNorm without re-centering, so computationally more efficient

    Args:
        model (torch.nn.Module): The model to modify in-place.
        eps (float, optional): numerical stability constant. Default: ``1e-8``.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional):
            Existing optimizers bound to ``model.parameters()``. All optimizers that have already been
            constructed with ``model.parameters()`` must be specified here so that
            they will optimize the correct parameters.

            If the optimizer(s) are constructed *after* calling this function,
            then it is safe to omit this parameter. These optimizers will see the correct
            model parameters.

    Returns:
        The modified model

    Example:
        .. testcode::

            import composer.functional as cf
            from torchvision import models
            model = models.resnet50()
            cf.apply_rms_norm(model)
    """

    def maybe_replace(module: torch.nn.Module, _: int) -> Optional[torch.nn.Module]:
        if isinstance(module, LayerNorm):
            return _RMSNorm.from_layer_norm(module, eps=eps)

    # we have to specify class names explicitly because replace_module_classes
    # now checks if `module.__class__ == cls`, rather than `isinstance(module, cls)`
    policy: Dict[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {LayerNorm: maybe_replace}
    module_surgery.replace_module_classes(model, optimizers=optimizers, policies=policy)
    return model


class RMSNorm(Algorithm):
    """Replaces layer normalization modules with
    `Root Mean Square LayerNorm <https://arxiv.org/abs/1910.07467>`_ modules
    which re-scale without re-centering, so requires fewer computations

    Runs on :attr:`.Event.INIT`.

    Args:
        eps (float, optional): size of value to add to denominator for numerical stability. Default: ``1e-8``.
    """

    def __init__(self, eps: float = 1e-8,):
        self.eps = eps

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(eps={self.eps})'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, _: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        assert state.model is not None, 'Model must be in state'

        apply_rms_norm(model=state.model, optimizers=state.optimizers)
        self._log_results(event, state, logger)

    def _log_results(self, _: Event, state: State, logger: Optional[Logger] = None) -> None:
        """Logs the result of RMSNorm applications, including the number of modules that have been replaced."""
        assert state.model is not None

        num_new_modules = module_surgery.count_module_instances(state.model, _RMSNorm)
        classname = 'RMSNorm'
        module_name = 'RMSNorm'

        # python logger
        log.info(f'Applied {classname} to model {state.model.__class__.__name__} '
                 f'with eps={self.eps}, '
                 f'Model now has {num_new_modules} {module_name} modules')

        if logger is not None:
            logger.log_hyperparameters({
                f'{classname}/num_new_modules': num_new_modules,
            })