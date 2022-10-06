# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Replaces all instances of `torch.nn.LayerNorm` with a `apex.normalization.fused_layer_norm.FusedRMSNorm
<https://nvidia.github.io/apex/layernorm.html>`_.

By fusing multiple kernel launches into one and using a simpler computation (RMSNorm vs LayerNorm), this usually improves GPU utilization.

See the :doc:`Method Card </method_cards/fused_rms_norm>` for more details.
"""

from composer.algorithms.fused_rms_norm.fused_rms_norm import FusedRMSNorm, apply_fused_rms_norm

__all__ = ['FusedRMSNorm', 'apply_fused_rms_norm']
