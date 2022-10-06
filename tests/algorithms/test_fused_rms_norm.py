# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import pytest
from torch.nn import LayerNorm

from composer.algorithms.fused_rms_norm import FusedRMSNorm, apply_fused_rms_norm
from composer.core.event import Event
from composer.loggers import Logger
from tests.common import device
from tests.fixtures.synthetic_hf_state import make_dataset_configs, synthetic_hf_state_maker


@pytest.fixture()
def synthetic_bert_state():
    synthetic_config = make_dataset_configs(model_family=['bert'])[0]
    return synthetic_hf_state_maker(synthetic_config)


def assert_is_frn_instance(model):
    pytest.importorskip('apex')
    pytest.importorskip('transformers')
    from apex.normalization.fused_layer_norm import FusedRMSNorm as APEXFusedRMSNorm
    from transformers import BertForMaskedLM, BertForSequenceClassification

    assert isinstance(model, BertForMaskedLM) or isinstance(model, BertForSequenceClassification)
    # ensure that within the entire model, no PyTorch LayerNorm exists, and at least one APEX FLN does.
    assert model.modules is not None, 'model has .modules method'
    for module_class in model.modules():
        assert not isinstance(
            module_class, LayerNorm), 'A torch.nn.LayerNorm should not be found in the model after surgery is applied.'

    assert any(isinstance(module_class, APEXFusedRMSNorm) for module_class in model.modules()
              ), 'apex.normalization.fused_layer_norm.FusedRMSNorm is not found in the post-surgery model.'


@device('gpu')
def test_fused_rms_norm_functional(synthetic_bert_state: Tuple, device: str):
    state, _, _ = synthetic_bert_state
    apply_fused_rms_norm(state.model, state.optimizers)
    assert_is_frn_instance(state.model.model)


@device('gpu')
def test_fused_rms_norm_algorithm(synthetic_bert_state: Tuple, empty_logger: Logger, device: str):
    pytest.importorskip('transformers')
    from transformers import BertForMaskedLM, BertForSequenceClassification

    state, _, _ = synthetic_bert_state
    fused_rms_norm = FusedRMSNorm()
    if device == 'gpu':
        state.model = state.model.cuda()  # move the model to gpu

    # state.model wrapped in HuggingFaceModel wrapped
    assert isinstance(state.model.model, BertForMaskedLM) or isinstance(state.model.model,
                                                                        BertForSequenceClassification)
    fused_rms_norm.apply(Event.INIT, state, empty_logger)

    assert_is_frn_instance(state.model.model)
