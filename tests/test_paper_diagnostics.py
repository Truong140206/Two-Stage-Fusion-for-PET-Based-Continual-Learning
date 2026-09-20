"""CPU-only tests for fixed diagnostic commands and overlap validation."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from tools import verify_paper_diagnostics as diag


class DiagnosticsTests(unittest.TestCase):
    def test_six_configurations_and_counter_branches(self):
        for ds in diag.verify.DATASETS:
            args = SimpleNamespace(dataset=ds, batch_size=24, port=29558,
                                   resolved_data_path=Path('/data'))
            for arm in diag.FIELDS:
                cmd = diag.command(args, arm, Path('/tii'), Path('/lora'))
                cfg = diag.grid.parsed_config(cmd)
                self.assertEqual(cfg.dataset, diag.verify.DATASETS[ds][0])
                self.assertTrue(cfg.eval and cfg.strict_exemplar_free and cfg.rp_pin_extractor)
                self.assertEqual((cfg.seed, cfg.num_tasks, cfg.rp_dim), (42, 10, 10000))
                self.assertEqual(cfg.rp_class_fusion_weight, 0)
                self.assertEqual(cfg.rp_class_fusion_gate, 'none')
                self.assertEqual(cfg.rp_route_audit, arm == 'rp_only')
                self.assertEqual(cfg.classifier_union_audit, arm == 'route_only')
                self.assertEqual(cfg.rp_route_fusion_drm, arm == 'route_only')
                self.assertEqual(cfg.rp_route_fusion_weight, 1 if arm == 'rp_only' else .7)
                self.assertEqual(cmd[:-1], diag.command(args, arm, Path('/tii'), Path('/lora'), False))

    def test_imr_route_reference_exact_command(self):
        args = SimpleNamespace(dataset='imr', batch_size=24, port=29558,
                               resolved_data_path=Path('/data'))
        self.assertEqual(diag.command(args, 'route_only', Path('/tii'), Path('/lora'), False),
                         diag.ab.command('sup', 'route_only', Path('/tii'), Path('/lora'), Path('/data')))

    def rows(self, arm):
        row = (dict(RouteTII=60., RouteRP=70., RouteUnion=80., RouteBoth=50.,
                    RouteAgree=75., RouteRPOnly=20., **{'Acc@task': 70.})
               if arm == 'rp_only' else
               dict(ClsRouted=74., ClsRP=70., ClsUnion=79., ClsRPOnly=5., **{'Acc@1': 74.}))
        return {i: row.copy() for i in range(1, 11)}

    def test_recovery_uses_task_balanced_denominator(self):
        result = diag.diagnostics(self.rows('rp_only'), 'rp_only')
        self.assertEqual(result['task_balanced_error_recovery_pct'], 50)
        self.assertEqual(result['task_balanced_tii_only_pp'], 10)

    def test_class_union_and_complement(self):
        result = diag.diagnostics(self.rows('route_only'), 'route_only')
        self.assertEqual(result['task_balanced_routed_only_pp'], 9)

    def test_missing_counter_rejected(self):
        rows = self.rows('rp_only')
        del rows[4]['RouteTII']
        with self.assertRaises(ValueError):
            diag.diagnostics(rows, 'rp_only')

    def test_invalid_and_inconsistent_counters_rejected(self):
        for value in (float('nan'), float('inf'), -1, 101, 72):
            rows = self.rows('route_only')
            rows[10]['ClsUnion'] = value
            with self.assertRaises(ValueError):
                diag.diagnostics(rows, 'route_only')

    def test_zero_errors_returns_no_ratio(self):
        rows = {i: dict(RouteTII=100., RouteRP=70., RouteUnion=100., RouteBoth=70.,
                        RouteAgree=70., RouteRPOnly=0., **{'Acc@task': 70.})
                for i in range(1, 11)}
        self.assertIsNone(diag.diagnostics(rows, 'rp_only')['task_balanced_error_recovery_pct'])

    def test_wrong_arm_rejected(self):
        with self.assertRaises(ValueError):
            diag.command(None, 'tune', None, None)

    def test_changed_historical_numbers_do_not_discard_new_measurement(self):
        pair = {arm: dict(diagnostics=diag.diagnostics(self.rows(arm), arm))
                for arm in diag.FIELDS}
        result = diag.compare_claims('imr', pair, {'Acc@task': 77.7901})
        self.assertEqual(result['DRM']['status'], 'MATCH_ROUNDING')
        self.assertEqual(result['RouteTII']['status'], 'UPDATE_PAPER')
        self.assertEqual(result['RouteTII']['measured'], 60)


if __name__ == '__main__':
    unittest.main()
