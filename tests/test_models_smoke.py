"""Model smoke tests: FEANet / FEANetBase / Classifier forward shapes (CPU, random inputs).

Skipped (not failed) when the ImageNet pretrained-weight download fails, so it also
runs offline.
"""
import unittest

import torch


def _make_input(b=2, size=128):
    return torch.randn(b, 3, size, size)


class TestModels(unittest.TestCase):
    def _try_forward(self, model, name):
        try:
            model.eval()
            with torch.no_grad():
                return model(_make_input())
        except Exception as e:
            msg = str(e).lower()
            if 'download' in msg or 'ssl' in msg or 'connect' in msg or 'timeout' in msg:
                self.skipTest(f'{name}: pretrained-weight download failed, skip ({e})')
            raise

    def test_feanet_base(self):
        from models.fea_net import FEANetBase
        m = FEANetBase()
        out = self._try_forward(m, 'FEANetBase')
        self.assertIsInstance(out, (tuple, list))
        self.assertEqual(out[0].shape[1], 2048)

    def test_feanet(self):
        from models.fea_net import FEANet
        m = FEANet()
        out = self._try_forward(m, 'FEANet')
        self.assertIsInstance(out, (tuple, list))
        self.assertEqual(out[0].shape[1], 2048)

    def test_classifier(self):
        from models.fea_net import Classifier
        clf = Classifier(2048, 3, prob=0.0)
        x = torch.randn(2, 2048)
        logit, mid = clf(x)
        self.assertEqual(logit.shape, (2, 3))
        self.assertEqual(mid.shape, (2, 1024))  # mid_out = intermediate 1024-D before the last layer

    def test_discriminator_baseline(self):
        from models.fea_net import DiscriminatorBaseline
        netD = DiscriminatorBaseline(input_dims=2048, hidden_dims=500, output_dims=2)
        out = netD(torch.randn(4, 2048))
        self.assertEqual(out.shape, (4, 2))


if __name__ == '__main__':
    unittest.main()
