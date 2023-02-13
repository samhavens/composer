# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Replaces batch normalization modules with `Root Mean Square Layer Normalization <https://arxiv.org/abs/1705.08741>`_ modules
that simulate the effect of using a smaller batch size.

See :class:`~composer.algorithms.RMSNorm` or the :doc:`Method Card </method_cards/rms_norm>` for details.
"""

from composer.algorithms.rms_norm.rms_norm import RMSNorm
from composer.algorithms.rms_norm.rms_norm import apply_rms_norm

__all__ = ['RMSNorm', 'apply_rms_norm']
