"""Exercise the actual evaluator with deterministic fake model outputs (CPU)."""
from types import SimpleNamespace
import pytest
import torch
from engines import hrm_lora_wtp_and_tap_engine as engine


class FixedModel(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.calls = 0

    def forward(self, inputs, **kwargs):
        logits = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return {'logits': torch.tensor(logits, dtype=torch.float32).repeat(len(inputs), 1),
                'pre_logits': inputs}


def evaluate_fake(monkeypatch, outputs, *, tau=1.1, fusion=False):
    args = SimpleNamespace(print_freq=100, train_mask=True, task_inc=False,
                           nb_classes=6, En='msp', tau=tau,
                           rp_route_fusion_drm=fusion, rp_feature_source='original',
                           rp_class_fusion_weight=1.0 if fusion else 0.0,
                           classifier_union_audit=fusion)
    monkeypatch.setattr(engine, 'args_ref', [None])
    monkeypatch.setattr(engine, 'gate_stats', {})
    original = FixedModel([[10, 0, 0, 0, 0, 0]])  # TII chooses task 0.
    model = FixedModel(outputs)
    loader = [(torch.zeros(1, 2), torch.tensor([2]))]
    if fusion:
        monkeypatch.setattr(engine, 'rp_extractor', lambda model, args: model)
        monkeypatch.setattr(engine, 'fuse_routers', lambda *a, **k: torch.tensor([1]))
        monkeypatch.setattr(engine, 'rp_head_predict',
                            lambda *a: torch.tensor([[0., 0., 20., 0., -float('inf'), -float('inf')]]))
        monkeypatch.setattr(engine, 'fuse_class_scores', lambda routed, rp, *a: rp)
    return engine.evaluate(model, original, loader, torch.device('cpu'), i=1, task_id=1,
                           class_mask=[[0, 1], [2, 3], [4, 5]],
                           target_task_map={i: i // 2 for i in range(6)}, args=args)


def test_drm_cannot_restore_unseen_classes(monkeypatch):
    stats = evaluate_fake(monkeypatch, [[1, 0, 5, 0, 0, 0],
                                        [1, 0, 5, 0, 100, 0],
                                        [1, 0, 5, 0, 100, 0]], tau=-1)
    assert stats['Acc@1'] == 100.0


def test_crm_masks_candidates_before_energy_selection(monkeypatch):
    # Correct DRM candidate: seen-class peak 5.
    # Alternative: seen-class peak 4, but unseen peak 100 must not win CRM.
    stats = evaluate_fake(monkeypatch, [[1, 0, 5, 0, 0, 0],
                                        [0, 0, 5, 0, 0, 0],
                                        [4, 0, 0, 0, 100, 0]])
    assert stats['Acc@1'] == 100.0


def test_classifier_audit_measures_prefusion_correctness(monkeypatch):
    stats = evaluate_fake(monkeypatch, [[1, 0, 5, 0, 0, 0],
                                        [5, 0, 1, 0, 0, 0],
                                        [5, 0, 1, 0, 0, 0]], fusion=True)
    assert stats['Acc@1'] == 100.0
    assert stats['ClsRouted'] == 0.0
    assert stats['ClsRP'] == stats['ClsUnion'] == stats['ClsRPOnly'] == 100.0


@pytest.mark.parametrize('seen', [1, 2, 3])
def test_mask_domain_and_final_stage_identity(seen):
    logits = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    args = SimpleNamespace(train_mask=True, task_inc=False)
    actual = engine._mask_evaluation_logits(logits, [[0, 1], [2, 3], [4, 5]], seen, args)
    assert torch.equal(actual[:, :2 * seen], logits[:, :2 * seen])
    assert torch.isneginf(actual[:, 2 * seen:]).all()
    if seen == 3:
        assert torch.equal(actual, logits)


def test_empty_batch_and_task_incremental_mask():
    args = SimpleNamespace(train_mask=True, task_inc=True)
    masks = [[0, 1], [2, 3], [4, 5]]
    empty = engine._mask_evaluation_logits(torch.empty(0, 6), masks, 2, args, 1)
    assert empty.shape == (0, 6)
    actual = engine._mask_evaluation_logits(torch.zeros(1, 6), masks, 2, args, 1)
    assert torch.isfinite(actual[:, 2:4]).all()
    assert torch.isneginf(actual[:, [0, 1, 4, 5]]).all()


def test_shuffled_class_ids_are_masked_by_membership_not_prefix():
    args = SimpleNamespace(train_mask=True, task_inc=False)
    actual = engine._mask_evaluation_logits(
        torch.zeros(1, 6), [[4, 1], [0, 5], [2, 3]], 1, args)
    assert torch.isfinite(actual[:, [4, 1]]).all()
    assert torch.isneginf(actual[:, [0, 2, 3, 5]]).all()
