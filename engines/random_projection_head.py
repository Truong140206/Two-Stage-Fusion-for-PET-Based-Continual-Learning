"""Routing-free second-order classifier head (RanPAC-style).

Motivation
----------
HRM-PET routes every query to one task LoRA before classifying, so its accuracy
is capped by task-inference quality. Measurements on Split-CUB200 showed that
ceiling is hard: exhaustive re-matching, Gaussian re-scoring and doubling the
LoRA rank all failed to move Acc@1 past ~86.5. Published exemplar-free results
(RanPAC, CUB B0I10) reach ~90 without any per-task routing, by classifying in a
high-dimensional random feature space with decorrelated second-order statistics.

This module implements that head:

    h = phi(f @ W)                       frozen random projection + nonlinearity
    G = sum_i h_i h_i^T                  Gram matrix  (M x M)
    C = sum_i h_i onehot(y_i)^T          class prototypes (M x nb_classes)
    s = h_test^T (G + lambda I)^-1 C     ridge / LDA-style scores

Only the aggregates G and C are retained -- no per-example features and no
images -- so the strict exemplar-free protocol is preserved (same category as
the class mean/covariance HRM-PET already stores). The projection is generated
from a fixed seed, so it never needs to be stored either.

The Gram inverse decorrelates the expanded features, which is what a per-class
diagonal/full covariance in the raw 768-d space failed to do.
"""

import math

import torch


_state = {
    'projection': None,
    'gram': None,
    'prototypes': None,
    # Second copy of the statistics over an 80% subset, so lambda can be chosen
    # against the held-out 20% instead of being a constant somebody tuned once.
    # The two halves are accumulated separately rather than one being derived by
    # subtracting the other from the running total: with rp_lambda_scope='task'
    # they are cleared every task while the running total is not, so the
    # subtraction identity no longer holds. See _select_lambda.
    'gram_fit': None,
    'prototypes_fit': None,
    'gram_val': None,
    'prototypes_val': None,
    # Held-out projected features of the CURRENT task, kept only when the
    # accuracy criterion is in use: a top-1 count cannot be recovered from
    # second-order statistics. Cleared at every task boundary.
    'val_features': None,
    'val_targets': None,
    'count': 0,
    'count_fit': 0,
    'count_val': 0,
    'split_index': 0,
    'dim': None,
    'seen_classes': set(),
}


def reset_rp_head():
    _state['projection'] = None
    _state['gram'] = None
    _state['prototypes'] = None
    _state['gram_fit'] = None
    _state['prototypes_fit'] = None
    _state['gram_val'] = None
    _state['prototypes_val'] = None
    _state['val_features'] = None
    _state['val_targets'] = None
    _state['count'] = 0
    _state['count_fit'] = 0
    _state['count_val'] = 0
    _state['split_index'] = 0
    _state['dim'] = None
    _state['seen_classes'] = set()
    _state['temperature'] = None
    _state['weights'] = None
    _state['lambda_used'] = None
    # solve_rp_head always rewrites this, so a stale mask has never actually
    # leaked. Clearing it anyway: the invariant that keeps it safe lives in
    # another function, and a reset should leave nothing behind.
    _state['class_mask_bias'] = None


def rp_head_ready():
    return _state['gram'] is not None and bool(_state['seen_classes'])


def get_rp_state():
    return _state


def _activation(x, kind):
    if kind == 'relu':
        return torch.relu(x)
    if kind == 'square':
        return x * x
    if kind == 'gelu':
        return torch.nn.functional.gelu(x)
    if kind == 'none':
        return x
    raise ValueError('Unknown rp_activation: {}'.format(kind))


def _ensure_projection(feature_dim, device, args):
    """Frozen random projection, regenerated deterministically from a seed."""
    if _state['projection'] is not None:
        return _state['projection']
    dim = int(getattr(args, 'rp_dim', 5000))
    seed = int(getattr(args, 'rp_seed', 1993))
    generator = torch.Generator(device='cpu').manual_seed(seed)
    projection = torch.randn(
        feature_dim, dim, generator=generator, dtype=torch.float32)
    projection = projection.to(device)
    _state['projection'] = projection
    _state['dim'] = dim
    return projection


def _normalize(features, args):
    """Feature scaling before projection.

    Raw ViT pre_logits have a large and uneven scale, which distorts a ReLU
    random projection and makes a single ridge lambda fit all classes badly.
    """
    kind = str(getattr(args, 'rp_normalize', 'none'))
    if kind == 'none':
        return features
    if kind == 'l2':
        return torch.nn.functional.normalize(features, dim=1)
    if kind == 'scale':
        return features / features.norm(dim=1, keepdim=True).mean().clamp_min(1e-6)
    raise ValueError('Unknown rp_normalize: {}'.format(kind))


def project_features(features, args, device):
    features = _normalize(features.float(), args)
    # rp_dim <= 0 skips the projection entirely: plain ridge on the raw
    # features, which is the paper's "no RP" reference point.
    if int(getattr(args, 'rp_dim', 5000)) <= 0:
        return features
    projection = _ensure_projection(features.shape[1], device, args)
    kind = str(getattr(args, 'rp_activation', 'relu'))
    return _activation(features @ projection, kind)


@torch.no_grad()
def accumulate_rp_statistics(features, targets, args, device, nb_classes):
    """Add one batch of training features to the Gram matrix and prototypes."""
    projected = project_features(features, args, device)
    dim = projected.shape[1]
    if _state['gram'] is None:
        _state['gram'] = torch.zeros(
            (dim, dim), dtype=torch.float64, device=device)
        _state['prototypes'] = torch.zeros(
            (dim, nb_classes), dtype=torch.float64, device=device)
    projected = projected.double()
    _state['gram'] += projected.T @ projected
    onehot = torch.zeros(
        (projected.shape[0], nb_classes), dtype=torch.float64, device=device)
    onehot[torch.arange(projected.shape[0], device=device),
           targets.to(device).long()] = 1.0
    _state['prototypes'] += projected.T @ onehot
    _state['seen_classes'].update(int(t) for t in targets.detach().cpu().tolist())

    # Mirror the accumulation over a fixed 80% subset. The split is a running
    # index rather than a random draw so a run is reproducible without carrying
    # another generator, and because the caller walks the data class by class it
    # comes out stratified for free: every class contributes its own fifth.
    batch = projected.shape[0]
    index = torch.arange(_state['split_index'], _state['split_index'] + batch,
                         device=device)
    _state['split_index'] += batch
    _state['count'] += batch
    if not bool(getattr(args, 'rp_lambda_search', False)):
        return
    if _state['gram_fit'] is None:
        _state['gram_fit'] = torch.zeros(
            (dim, dim), dtype=torch.float64, device=device)
        _state['prototypes_fit'] = torch.zeros(
            (dim, nb_classes), dtype=torch.float64, device=device)
        _state['gram_val'] = torch.zeros(
            (dim, dim), dtype=torch.float64, device=device)
        _state['prototypes_val'] = torch.zeros(
            (dim, nb_classes), dtype=torch.float64, device=device)
    keep = (index % 5) != 4
    if bool(keep.any()):
        fit_projected = projected[keep]
        _state['gram_fit'] += fit_projected.T @ fit_projected
        _state['prototypes_fit'] += fit_projected.T @ onehot[keep]
        _state['count_fit'] += int(keep.sum())
    held = ~keep
    if bool(held.any()):
        val_projected = projected[held]
        _state['gram_val'] += val_projected.T @ val_projected
        _state['prototypes_val'] += val_projected.T @ onehot[held]
        _state['count_val'] += int(held.sum())
        if str(getattr(args, 'rp_lambda_criterion', 'mse')) == 'accuracy':
            # One fifth of one task at rp_dim=10000 is tens of megabytes, so
            # this is cheap; the point of the closed form was elegance, not a
            # memory bound. Only the current task is ever held.
            if _state['val_features'] is None:
                _state['val_features'] = []
                _state['val_targets'] = []
            _state['val_features'].append(val_projected.clone())
            _state['val_targets'].append(targets.to(device).long()[held])


def begin_rp_task(args):
    """Open a fresh held-out split for the task about to be accumulated.

    This is the difference between running RanPAC's procedure and running
    something adjacent to it. RanPAC calls its ridge search with the *current
    task's* features and rebuilds the 80:20 split from scratch every task; the
    accumulated Gram it finally solves with is a separate, larger object. We
    previously let the 80:20 statistics accumulate across the whole sequence
    too, so by task ten the sweep was ranging over ten tasks pooled.

    The suspicion was that this biases the choice upward: the norm of a Gram
    matrix grows linearly with the number of samples in it, and the ridge that
    best balances fit against regularisation grows with that norm. That law is
    real -- test_pooling_the_sweep_biases_lambda_upward pins it exactly -- but
    it is the wrong model for pooling *tasks*. It applies when the same classes
    gain more samples. A new task brings its own new classes, so samples per
    class stays fixed while the total grows, and the optimal ridge tracks
    samples per class.

    Measured on ImageNet-R seed 42: both scopes select 1e5 at all ten tasks and
    give bit-identical final metrics. So the divergence was real and this is the
    faithful version, but it changes no number we report. Keep it anyway --
    matching the method we build on is worth more than the diff it saves, and
    'pooled' is what makes the two comparable rather than arguable.

    Only the fit/validation copies are cleared. The running `gram` and
    `prototypes` -- the ones the head is actually solved from -- keep
    accumulating, because their order invariance is what makes the head immune
    to forgetting.
    """
    if not bool(getattr(args, 'rp_lambda_search', False)):
        return

    # Unconditional, and deliberately before the scope check. Held-out FEATURES
    # are per-sample data; carrying them past a task boundary would be retaining
    # old-task examples, which the exemplar-free protocol forbids outright. The
    # scope setting decides how much *statistics* the sweep sees, and statistics
    # are not examples -- it must never be able to authorise keeping features.
    #
    # This is also the better proxy, not merely the legal one. With pooled
    # statistics the ridge solution spans every class seen so far, so scoring
    # the current task's held-out fifth against all seen classes measures the
    # problem the deployed head actually faces. Scoping both to one task
    # measures a 20-way problem while deployment is up to 200-way, and the best
    # regularisation for one is not the best for the other.
    _state['val_features'] = None
    _state['val_targets'] = None

    if str(getattr(args, 'rp_lambda_scope', 'task')) != 'task':
        return
    for key in ('gram_fit', 'prototypes_fit', 'gram_val', 'prototypes_val'):
        if _state[key] is not None:
            # In place: at rp_dim=10000 each of these is 800 MB and reallocating
            # four of them every task would churn 3.2 GB for nothing.
            _state[key].zero_()
    _state['count_fit'] = 0
    _state['count_val'] = 0
    _state['split_index'] = 0


def _ridge(gram, lam):
    """G + lambda I, adding along the diagonal instead of building lam * I.

    At rp_dim=10000 a float64 identity is 800 MB and `lam * eye` materialises a
    second one, so the obvious form spends 2.4 GB of temporaries to express
    something that touches 10000 numbers. Arithmetically identical: every
    off-diagonal entry was gram + 0.0. This matters more now that the lambda
    search calls it seventeen times per task.
    """
    out = gram.clone()
    out.diagonal().add_(lam)
    return out


def _ridge_solve(gram, prototypes, lam):
    chol = torch.linalg.cholesky(_ridge(gram, lam))
    return torch.cholesky_solve(prototypes, chol)


@torch.no_grad()
def _select_lambda(args, device):
    """Choose lambda on held-out data from the current task, RanPAC's procedure.

    RanPAC does not ship a fixed lambda. At every task it splits that task's own
    data 80:20, sweeps lambda across 17 orders of magnitude, and keeps whichever
    value minimises squared error on the held-out fifth. Old tasks are never
    touched, so the continual-learning constraint holds. We had replaced that
    step with a constant tuned once on ImageNet-R seed 42 -- which is both a
    deviation from the method we build on and the single largest source of
    hyperparameter selection bias in the report.

    The validation error needs no stored features. With one-hot targets,

        ||Y - W^T H||_F^2 = n_val - 2 tr(W^T C_val) + tr(W^T G_val W)

    and the held-out statistics are just the difference between the full and the
    fitted accumulators. So the sweep is closed-form in quantities we already
    keep, and the exemplar-free protocol is preserved by construction rather
    than by argument.

    The sweep is the expensive part -- 17 Cholesky factorisations of a dim x dim
    matrix per task. RanPAC's own write-up names the same step as its bottleneck.

    Two criteria, because RanPAC's own one is wrong for how we use the scores.

    'mse' is theirs: minimise the held-out squared error. Measured on ImageNet-R
    seed 42 it picks 1e5 at every one of the ten tasks and lands on Acc@1
    75.0795, against 75.5116 for the hand-tuned 1e4 -- the worst of the five
    values in our own lambda sweep.

    The scope question is settled: rerunning under scope='task', which is what
    RanPAC actually does, selects 1e5 at all ten tasks as well and lands on the
    same 75.0795. How much data the sweep sees is not what drives the choice.

    What remains is the criterion itself. Y is one-hot, so shrinking W toward
    zero shrinks the residual, and squared error therefore rewards a magnitude
    shrinkage that argmax ignores. In RanPAC the ridge scores are the final
    classifier, so magnitude is part of the answer and the criterion fits. Here
    they are standardised per sample and blended into a routed head, which
    discards magnitude entirely -- so the criterion optimises a quantity the
    next step throws away. Note this is not refuted by 'cosine' agreeing: what
    the head needs is complementarity with the routed classifier, and neither
    criterion measures that. Both measure fit.

    'cosine' fixes exactly that by being invariant to the scale of W:

        cos = tr(W^T C_val) / sqrt( tr(W^T G_val W) * n_val )

    which is the Frobenius cosine between the predictions HW and the targets Y.
    Multiply W by any positive constant and numerator and denominator move
    together. Still closed-form in the statistics already kept, so it costs
    nothing extra and stores no features.
    """
    gram_fit = _state.get('gram_fit')
    if gram_fit is None:
        raise RuntimeError(
            'rp_lambda_search is on but no 80% statistics were accumulated; '
            'accumulate_rp_statistics must run with the same flag set')
    n_val = int(_state['count_val'])
    if n_val <= 0:
        raise RuntimeError('rp_lambda_search has an empty validation split')

    prototypes_fit = _state['prototypes_fit']
    gram_val = _state['gram_val']
    prototypes_val = _state['prototypes_val']

    criterion = str(getattr(args, 'rp_lambda_criterion', 'mse'))
    if criterion not in ('mse', 'cosine', 'accuracy'):
        raise ValueError('unknown rp_lambda_criterion: %s' % criterion)

    val_x, val_y, seen_index = None, None, None
    if criterion == 'accuracy':
        if not _state.get('val_features'):
            raise RuntimeError(
                "rp_lambda_criterion='accuracy' but no held-out features were "
                'kept; accumulate_rp_statistics must run with the same setting')
        val_x = torch.cat(_state['val_features'], dim=0)
        val_y = torch.cat(_state['val_targets'], dim=0)
        # Score only over classes seen so far, mirroring what the deployed head
        # does. Ranking against classes that cannot occur would let a lambda be
        # rewarded or punished for predictions the system can never make.
        seen_index = torch.tensor(
            sorted(int(c) for c in _state['seen_classes']),
            device=val_x.device, dtype=torch.long)

    # _ridge_solve clones the Gram to add lambda, which at rp_dim=10000 is a
    # 800 MB temporary -- seventeen times per task, on top of the factorisation
    # itself. That is what killed a bare-extractor run at task 4. Writing the
    # diagonal in place removes the clone. Restoring it by copying a saved copy
    # rather than subtracting lambda back off keeps the accumulator bit-exact:
    # (x + lam) - lam is not x in floating point once lam dominates x, and this
    # matrix has to survive the whole task sequence.
    saved_diagonal = gram_fit.diagonal().clone()

    best_lambda, best_score = None, None
    tried = []
    try:
        for exponent in range(-8, 9):
            lam = float(10.0 ** exponent)
            gram_fit.diagonal().copy_(saved_diagonal).add_(lam)
            try:
                chol = torch.linalg.cholesky(gram_fit)
                weights = torch.cholesky_solve(prototypes_fit, chol)
                del chol
            except torch.linalg.LinAlgError:
                # Tiny lambda on a rank-deficient Gram. Not a failure, just a
                # value the data cannot support; skip it rather than abort.
                continue
            if criterion == 'accuracy':
                predicted = seen_index[
                    (val_x @ weights)[:, seen_index].argmax(dim=1)]
                score = float((predicted == val_y).double().mean())
            else:
                agreement = float((weights * prototypes_val).sum())    # <HW, Y>
                energy = float((weights * (gram_val @ weights)).sum())  # ||HW||^2
                if criterion == 'mse':
                    score = -(float(n_val) - 2.0 * agreement + energy)
                else:
                    score = agreement / math.sqrt(
                        max(energy * float(n_val), 1e-30))
            tried.append((lam, score))
            if best_score is None or score > best_score:
                best_lambda, best_score = lam, score
            del weights
    finally:
        # Unconditional: an out-of-memory escaping mid-sweep would otherwise
        # leave a stray lambda on the diagonal, and every later task would build
        # on a corrupted accumulator without any error to show for it.
        gram_fit.diagonal().copy_(saved_diagonal)
    # gram_val and prototypes_val are state now, not temporaries built here, so
    # there is nothing to free.

    if best_lambda is None:
        raise RuntimeError('rp_lambda_search: every lambda failed to solve')
    curve = ' '.join('%.0e:%.4f' % (lam, value) for lam, value in tried)
    print('RP head lambda search [%s, scope=%s]: chose %.3g over %d values '
          '(%d fit / %d val samples)\n  curve: %s'
          % (criterion, str(getattr(args, 'rp_lambda_scope', 'task')),
             best_lambda, len(tried), _state['count_fit'], n_val, curve))
    return best_lambda


@torch.no_grad()
def solve_rp_head(args, device, seen_class_ids=None):
    """Ridge solve  Wout = (G + lambda I)^-1 C  for the current statistics."""
    gram = _state['gram']
    prototypes = _state['prototypes']
    if gram is None:
        raise RuntimeError('RP head has no accumulated statistics')
    if bool(getattr(args, 'rp_lambda_search', False)):
        # Selected on 80%, but the final weights are refitted on everything.
        # Holding out a fifth is a device for picking lambda, not a reason to
        # throw that fifth away once lambda is fixed.
        lam = _select_lambda(args, device)
    else:
        lam = float(getattr(args, 'rp_lambda', 1e4))
    _state['lambda_used'] = lam
    ridge = _ridge(gram, lam)
    try:
        chol = torch.linalg.cholesky(ridge)
        weights = torch.cholesky_solve(prototypes, chol)
    except torch.linalg.LinAlgError as err:
        # Only the not-positive-definite case falls back. A bare `except
        # Exception` here also swallowed OOM, and swallowed it into lstsq,
        # which needs more memory than the Cholesky it was replacing. Say so
        # rather than degrading in silence.
        print('RP head: Cholesky failed (', err, ') - falling back to lstsq. '
              'A singular Gram at lambda =', lam, 'means the statistics are '
              'degenerate; check that accumulation actually ran.')
        weights = torch.linalg.lstsq(ridge, prototypes).solution
    weights = weights.float()
    if seen_class_ids is not None:
        mask = torch.full((weights.shape[1],), float('-inf'), device=device)
        mask[torch.as_tensor(
            sorted(int(c) for c in seen_class_ids), device=device)] = 0.0
        _state['class_mask_bias'] = mask
    else:
        _state['class_mask_bias'] = None
    _state['weights'] = weights
    return weights


def fit_rp_temperature(scores, targets, args, device):
    """Fit one scalar temperature so the ridge scores read as calibrated logits.

    Ridge outputs are unnormalized regression values, so a cross-entropy loss
    computed on them is not comparable to a softmax classifier's loss. Only the
    scalar is kept; the scores used to fit it are transient.
    """
    log_temperature = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=60)
    # Unseen classes carry a -inf mask bias; differentiating through it yields
    # NaN gradients, so fit on finite scores.
    scores = torch.nan_to_num(
        scores.detach().to(device).float(),
        nan=0.0, posinf=1e4, neginf=-1e4)
    targets = targets.detach().to(device)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            scores / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().item())
    if not math.isfinite(temperature) or temperature <= 0.0:
        print('RP head calibration produced', temperature,
              '- falling back to temperature 1.0')
        temperature = 1.0
    _state['temperature'] = temperature
    return temperature


@torch.no_grad()
def rp_head_predict(features, args, device):
    """Score every class for a batch of backbone features (no routing)."""
    weights = _state.get('weights')
    if weights is None:
        raise RuntimeError('RP head weights not solved yet')
    projected = project_features(features, args, device)
    scores = projected @ weights
    temperature = _state.get('temperature')
    if temperature and math.isfinite(temperature) and temperature > 0.0:
        scores = scores / temperature
    bias = _state.get('class_mask_bias')
    if bias is not None:
        scores = scores + bias
    return scores
