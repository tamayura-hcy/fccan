"""Util smoke tests: every module imports and its core functions run (no real data/GPU).

Run from the project root:  python -m unittest tests.test_util_smoke
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch


class TestEmLoss(unittest.TestCase):
    def test_entropy_loss(self):
        from util.em_loss import entropy_loss, entropy_loss_masked, entropy_loss_weighted
        logits = torch.randn(8, 3)
        loss = entropy_loss(logits)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        # empty batch returns 0
        self.assertEqual(entropy_loss(torch.randn(0, 3)).item(), 0.0)
        # masked / weighted
        mask = torch.tensor([True, False] * 4)
        self.assertTrue(torch.isfinite(entropy_loss_masked(logits, mask)))
        w = torch.rand(8) + 0.1
        self.assertTrue(torch.isfinite(entropy_loss_weighted(logits, w)))


class TestCacoLoss(unittest.TestCase):
    def test_caco_catnce_loss(self):
        from util.caco_loss import caco_catnce_loss
        torch.manual_seed(0)
        q = torch.randn(8, 16)
        keys = torch.randn(24, 16)
        key_labels = torch.randint(0, 3, (24,))
        q_labels = torch.randint(0, 3, (8,))
        loss = caco_catnce_loss(q, keys, key_labels, q_labels, 3, tau=0.07)
        self.assertTrue(torch.isfinite(loss))
        # empty keys return 0
        self.assertEqual(caco_catnce_loss(q, torch.empty(0, 16), key_labels[:0], q_labels, 3).item(), 0.0)


class TestAngModule(unittest.TestCase):
    def test_angular_balance_loss(self):
        from util.ang import AngModule
        m = AngModule(num_classes=3, feat_dim=16)
        # near-equiangular unit vectors -> finite loss
        means = torch.tensor([[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 1., 0.]])
        loss = m.angular_balance_loss(means)
        self.assertTrue(torch.isfinite(loss))
        # only 1 class returns 0
        self.assertEqual(m.angular_balance_loss(torch.randn(1, 16)).item(), 0.0)


class TestLrSchedules(unittest.TestCase):
    def test_linear_decay(self):
        from util.lr_schedules import linear_decay_scale, late_linear_scale, apply_lr_scale
        self.assertAlmostEqual(linear_decay_scale(0, 10, 0.1), 1.0)
        self.assertAlmostEqual(linear_decay_scale(9, 10, 0.1), 0.1, places=6)
        self.assertEqual(linear_decay_scale(0, 1, 0.1), 1.0)
        # late_linear: full lr in the early phase
        self.assertEqual(late_linear_scale(0, 10, 0.5, 0.1), 1.0)
        # apply_lr_scale
        opt = torch.optim.SGD([torch.nn.Parameter(torch.randn(2))], lr=0.1)
        apply_lr_scale(opt, [0.1], 0.5)
        self.assertAlmostEqual(opt.param_groups[0]['lr'], 0.05)


class TestDataUtils(unittest.TestCase):
    def test_ensure_save_dir(self):
        from util.data_utils import ensure_save_dir, TASK_LIST
        tmp = tempfile.mkdtemp()
        try:
            sub = os.path.join(tmp, 'a', 'b')
            ensure_save_dir(sub)
            self.assertTrue(os.path.isdir(sub))
            ensure_save_dir('')  # empty path must not fail
            ensure_save_dir('.')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(len(TASK_LIST), 6)


class TestEvalUtils(unittest.TestCase):
    def test_heatmap(self):
        from util.eval_utils import heatmap
        y = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2])
        p = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2])
        img = heatmap(p, y)
        # seaborn present -> RGBA image; missing -> (1,1) empty array (not a failure)
        if img.ndim == 3:
            self.assertEqual(img.shape[-1], 4)


class TestDiagRuntime(unittest.TestCase):
    def test_diag_runtime(self):
        from util.diag_runtime import diag_v2_is_enabled, diag_v2_collector
        self.assertIsNone(diag_v2_collector())
        self.assertFalse(diag_v2_is_enabled())


class TestEnergyUda(unittest.TestCase):
    def test_energy_uda(self):
        from util.energy_uda import EnergyUdaState, logits_to_energy
        st = EnergyUdaState(num_classes=3, ema_momentum=0.1)
        logits = torch.randn(8, 3)
        e = logits_to_energy(logits)  # -logits, shape [B, C]
        self.assertEqual(e.shape, logits.shape)
        self.assertTrue(torch.isfinite(e).all().item())
        self.assertEqual(st.C, 3)


if __name__ == '__main__':
    unittest.main()
