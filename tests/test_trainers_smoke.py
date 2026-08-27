"""Trainer smoke tests: run 1 epoch on tiny random batches to verify the training loops don't crash.

No real dataset/GPU needed; skipped when pretrained-weight download fails.
"""
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset


def _make_dataset(n_per_class=3, size=96, num_classes=3):
    xs, ys = [], []
    for c in range(num_classes):
        for _ in range(n_per_class):
            xs.append(torch.randn(3, size, size))
            ys.append(c)
    return TensorDataset(torch.stack(xs), torch.tensor(ys))


def _make_loaders(batch=3):
    train_dl = DataLoader(_make_dataset(), batch_size=batch, shuffle=True)
    val_dl = DataLoader(_make_dataset(n_per_class=2), batch_size=batch, shuffle=False)
    return train_dl, val_dl


class TestTrainers(unittest.TestCase):
    def _model_factory(self, use_baseline=False):
        try:
            from models.fea_net import FEANet, FEANetBase, Classifier
            enc = FEANetBase() if use_baseline else FEANet()
            clf = Classifier(enc.combined_features, 3, prob=0.0)
            # The training loop moves data with .cuda(); the model must follow the device
            if torch.cuda.is_available():
                enc = enc.cuda()
                clf = clf.cuda()
            return enc, clf
        except Exception as e:
            msg = str(e).lower()
            if 'download' in msg or 'ssl' in msg or 'connect' in msg or 'timeout' in msg:
                self.skipTest(f'pretrained-weight download failed, skip ({e})')
            raise

    def test_train_src(self):
        from trainers.source_trainer import train_src
        enc, clf = self._model_factory(use_baseline=False)
        train_dl, val_dl = _make_loaders()
        with tempfile.TemporaryDirectory() as tmp:
            enc, clf, _ = train_src(
                enc, clf, train_dl, val_dl, epochs=1, save_name=tmp,
                src_optimizer='sgd', src_use_lr_sched=False,
                num_classes=3, src_ll_aug=0.0)
            self.assertIsNotNone(enc)

    def test_train_src_baseline(self):
        from trainers.source_trainer import train_src_baseline_ref
        enc, clf = self._model_factory(use_baseline=True)
        train_dl, val_dl = _make_loaders()
        with tempfile.TemporaryDirectory() as tmp:
            enc, clf, _ = train_src_baseline_ref(
                enc, clf, train_dl, val_dl, epochs=1, save_name=tmp)
            self.assertIsNotNone(enc)

    def test_train_tgt_caco(self):
        from trainers.target_trainer import train_tgt_caco
        src_enc, clf = self._model_factory(use_baseline=False)
        tgt_enc, _ = self._model_factory(use_baseline=False)
        train_dl, _ = _make_loaders()
        with tempfile.TemporaryDirectory() as tmp:
            tgt_enc, clf, _, accs, _ema_accs = train_tgt_caco(
                src_enc, clf, tgt_enc, train_dl, train_dl, tmp,
                num_epochs=1, num_classes=3,
                class_names=['AMD', 'DME', 'Normal'],
                lambda_em=1.0, lambda_caco=0.1,
                tgt_test_loader=None, tgt_test_size=None,
                ema_teacher=0.0, ema_guide_caco=0.0, use_energy_uda=False)
            self.assertIsNotNone(tgt_enc)

    def test_train_tgt_caco_lr_sched(self):
        """Advanced LR-schedule smoke: warmup_cosine survives 1 epoch (scale>0 during warmup)."""
        from trainers.target_trainer import train_tgt_caco
        src_enc, clf = self._model_factory(use_baseline=False)
        tgt_enc, _ = self._model_factory(use_baseline=False)
        train_dl, _ = _make_loaders()
        with tempfile.TemporaryDirectory() as tmp:
            tgt_enc, clf, _, accs, _ema_accs = train_tgt_caco(
                src_enc, clf, tgt_enc, train_dl, train_dl, tmp,
                num_epochs=1, num_classes=3,
                class_names=['AMD', 'DME', 'Normal'],
                lambda_em=1.0, lambda_caco=0.1,
                tgt_test_loader=None, tgt_test_size=None,
                tgt_lr_sched='warmup_cosine', tgt_lr_warmup_epochs=3,
                tgt_linear_eta_min_ratio=0.1,
                use_energy_uda=False)
            self.assertIsNotNone(tgt_enc)

    def test_train_tgt_caco_swa(self):
        """SWA smoke: swa_start covers the last epoch, survives 1 epoch."""
        from trainers.target_trainer import train_tgt_caco
        src_enc, clf = self._model_factory(use_baseline=False)
        tgt_enc, _ = self._model_factory(use_baseline=False)
        train_dl, _ = _make_loaders()
        with tempfile.TemporaryDirectory() as tmp:
            tgt_enc, clf, _, accs, _ema_accs = train_tgt_caco(
                src_enc, clf, tgt_enc, train_dl, train_dl, tmp,
                num_epochs=1, num_classes=3,
                class_names=['AMD', 'DME', 'Normal'],
                lambda_em=1.0, lambda_caco=0.1,
                tgt_test_loader=None, tgt_test_size=None,
                swa=1, swa_start_epoch=1,
                use_energy_uda=False)
            self.assertIsNotNone(tgt_enc)


if __name__ == '__main__':
    unittest.main()
