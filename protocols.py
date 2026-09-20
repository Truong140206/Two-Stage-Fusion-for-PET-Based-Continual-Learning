"""Protocol validation for rehearsal-free continual-learning experiments."""

import re
from types import SimpleNamespace


_FORBIDDEN_BOOLEAN_FLAGS = {
    'crct_real_feature_replay': 'stores per-example real features',
    'prototype_rematching': 'uses stored real-feature prototypes',
    'shared_prototype_router': 'reconstructs prototypes from historical train images',
    'replay_logit_calibration': 'fits calibration on retained real features',
    'replay_task_router': 'trains a router on retained real features',
    'calibrated_progressive_rematching': 'trains halting gates on historical train images',
    'distilled_router_rematching': 'trains a router on historical train images',
}


def exemplar_free_violations(args):
    violations = []
    for name, reason in _FORBIDDEN_BOOLEAN_FLAGS.items():
        if bool(getattr(args, name, False)):
            violations.append(f'--{name}: {reason}')
    if float(getattr(args, 'exhaustive_local_prototype_weight', 0.0)) > 0.0:
        violations.append(
            '--exhaustive_local_prototype_weight: uses stored real-feature prototypes')
    return violations


def validate_exemplar_free_protocol(args):
    if not bool(getattr(args, 'strict_exemplar_free', False)):
        return
    violations = exemplar_free_violations(args)
    if violations:
        details = '\n  - '.join(violations)
        raise ValueError(
            'Strict exemplar-free protocol rejected this configuration:\n  - '
            + details)
    print(
        'Strict exemplar-free protocol: PASS '
        '(no historical images or per-example real features).')


def exemplar_free_log_violations(text):
    """Recover protocol flags from a printed argparse Namespace."""
    if 'Namespace(' not in text:
        raise ValueError('Training log does not contain a Namespace configuration')

    values = {}
    for name in _FORBIDDEN_BOOLEAN_FLAGS:
        match = re.search(rf'\b{re.escape(name)}=(True|False)\b', text)
        values[name] = bool(match and match.group(1) == 'True')

    weight_match = re.search(
        r'\bexhaustive_local_prototype_weight=([-+0-9.eE]+)\b', text)
    values['exhaustive_local_prototype_weight'] = (
        float(weight_match.group(1)) if weight_match else 0.0)
    return exemplar_free_violations(SimpleNamespace(**values))


def validate_exemplar_free_training_log(path):
    """Fail closed when a checkpoint's training log violates the protocol."""
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    violations = exemplar_free_log_violations(text)
    if violations:
        details = '\n  - '.join(violations)
        raise ValueError(
            'Checkpoint training protocol is not exemplar-free:\n  - ' + details)
    print(
        'Checkpoint training protocol: PASS '
        '(no forbidden historical-image or per-example-feature mechanism).')
