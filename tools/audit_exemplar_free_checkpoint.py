#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protocols import exemplar_free_violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    violations = []
    if checkpoint.get('real_feature_memory'):
        violations.append('checkpoint contains real_feature_memory')
    saved_args = checkpoint.get('args')
    if saved_args is not None:
        violations.extend(exemplar_free_violations(saved_args))
        if not bool(getattr(saved_args, 'strict_exemplar_free', False)):
            violations.append('checkpoint was not created in strict mode')
    else:
        violations.append('checkpoint has no saved args to audit')

    if violations:
        raise SystemExit(
            'EXEMPLAR_FREE_AUDIT=FAIL\n  - ' + '\n  - '.join(violations))
    print('EXEMPLAR_FREE_AUDIT=PASS', args.checkpoint)


if __name__ == '__main__':
    main()
