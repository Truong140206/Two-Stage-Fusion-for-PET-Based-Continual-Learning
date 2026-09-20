from types import SimpleNamespace

import pytest

from protocols import (
    exemplar_free_log_violations,
    validate_exemplar_free_protocol,
    validate_exemplar_free_training_log,
)


def test_strict_protocol_accepts_paper_configuration():
    args = SimpleNamespace(strict_exemplar_free=True)
    validate_exemplar_free_protocol(args)


@pytest.mark.parametrize('flag', [
    'crct_real_feature_replay',
    'prototype_rematching',
    'shared_prototype_router',
    'replay_logit_calibration',
    'replay_task_router',
    'calibrated_progressive_rematching',
    'distilled_router_rematching',
])
def test_strict_protocol_rejects_historical_data_features(flag):
    args = SimpleNamespace(strict_exemplar_free=True, **{flag: True})
    with pytest.raises(ValueError, match=flag):
        validate_exemplar_free_protocol(args)


def test_strict_protocol_rejects_local_real_prototypes():
    args = SimpleNamespace(
        strict_exemplar_free=True,
        exhaustive_local_prototype_weight=0.1,
    )
    with pytest.raises(ValueError, match='exhaustive_local_prototype_weight'):
        validate_exemplar_free_protocol(args)


def test_training_log_audit_accepts_paper_configuration(tmp_path):
    log_path = tmp_path / 'baseline.log'
    log_path.write_text(
        'Namespace(crct_real_feature_replay=False, prototype_rematching=False, '
        'shared_prototype_router=False, replay_logit_calibration=False, '
        'replay_task_router=False, calibrated_progressive_rematching=False, '
        'distilled_router_rematching=False, '
        'exhaustive_local_prototype_weight=0.0)',
        encoding='utf-8',
    )
    assert exemplar_free_log_violations(log_path.read_text()) == []
    validate_exemplar_free_training_log(log_path)


def test_training_log_audit_rejects_real_feature_replay(tmp_path):
    log_path = tmp_path / 'feature_memory.log'
    log_path.write_text(
        'Namespace(crct_real_feature_replay=True, '
        'exhaustive_local_prototype_weight=0.0)',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='crct_real_feature_replay'):
        validate_exemplar_free_training_log(log_path)


def test_training_log_audit_fails_closed_without_namespace(tmp_path):
    log_path = tmp_path / 'incomplete.log'
    log_path.write_text('training started', encoding='utf-8')
    with pytest.raises(ValueError, match='does not contain a Namespace'):
        validate_exemplar_free_training_log(log_path)
