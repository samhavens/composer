# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock
import pytest
import torch
from composer.algorithms.rms_norm import RMSNorm, apply_rms_norm
from composer.algorithms.rms_norm.rms_norm import _RMSNorm
from composer.algorithms.warnings import NoEffectWarning
from composer.core.event import Event
from composer.core.state import State
from composer.models.tasks.classification import ComposerClassifier
from composer.utils import module_surgery


class SimpleLNModel(ComposerClassifier):
    """Small classification model with LayerNorm

    Args:
        embedding_dim (int): number of input features (default: 2)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, embedding_dim: int = 2, num_classes: int = 2, hidden_dim: int = 5) -> None:

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        fc1 = torch.nn.Linear(embedding_dim, hidden_dim)
        fc2 = torch.nn.Linear(hidden_dim, num_classes)
        layer_norm = torch.nn.LayerNorm(hidden_dim)

        net = torch.nn.Sequential(
            fc1,
            layer_norm,
            torch.nn.ReLU(),
            fc2,
            torch.nn.Softmax(dim=-1),
        )
        super().__init__(module=net)


@pytest.fixture
def algo_instance():
    return RMSNorm()


@pytest.fixture()
def layer_norm_instance():
    return SimpleLNModel()


@pytest.fixture
def state(minimal_state: State):
    minimal_state.model = SimpleLNModel()
    return minimal_state


def test_rms_norm_noeffectwarning():
    model = torch.nn.Linear(in_features=16, out_features=32)
    with pytest.warns(NoEffectWarning):
        apply_rms_norm(model)


def test_correct_event_matches(algo_instance):
    assert algo_instance.match(Event.INIT, Mock(side_effect=ValueError))


@pytest.mark.parametrize('event', Event)  # enum iteration
def test_incorrect_event_does_not_match(event: Event, algo_instance):
    if event == Event.INIT:
        return
    assert not algo_instance.match(event, Mock(side_effect=ValueError))


def test_algorithm_logging(state, algo_instance):
    logger_mock = Mock()
    algo_instance.apply(Event.INIT, state, logger_mock)
    logger_mock.log_hyperparameters.assert_called_once_with({
        'RMSNorm/num_new_modules': 1,
    })


def test_layer_norm_gets_replaced_functional(layer_norm_instance):
    assert (
        module_surgery.count_module_instances(layer_norm_instance, torch.nn.LayerNorm) == 1
        and module_surgery.count_module_instances(layer_norm_instance, _RMSNorm) == 0
    ), "Before surgery there is 1 LayerNorm layer and no RMSNorm layer"
    apply_rms_norm(layer_norm_instance)
    assert (
        module_surgery.count_module_instances(layer_norm_instance, torch.nn.LayerNorm) == 0
        and module_surgery.count_module_instances(layer_norm_instance, _RMSNorm) == 1
    ), "After surgery LayerNorms have been replaced with RMSNorm"
