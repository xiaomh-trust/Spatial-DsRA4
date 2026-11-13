"""Run: python -m unittest discover -s tests -p test_spatial_dsra_v4.py -v"""
import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
PATHS = [ROOT / 'openvla-SF/prismatic/models/spatial_dsra.py',
         ROOT / 'openpi-SF/src/openpi/models_pytorch/spatial_dsra.py']


def load(path):
    spec = importlib.util.spec_from_file_location('dsra_test_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpatialDsRATest(unittest.TestCase):
    # [Spatial-DsRA v4]: Exercise both independent packages with the same mathematical contract.
    def setUp(self):
        torch.manual_seed(7)

    def test_uniform_and_explicit_diagonal_formula(self):
        for path in PATHS:
            with self.subTest(path=str(path)):
                m = load(path).SpatialDsRAProjector(12, 20, 4, 8, 3)
                hidden = [torch.randn(2, 6, 12) for _ in range(4)]
                b = [torch.nn.functional.gelu(m.down(m.norm(h))) for h in hidden]
                mean = torch.stack(b).mean(0)
                torch.testing.assert_close(m.fuse(hidden), mean)
                with torch.no_grad():
                    m.rho.normal_()
                pi = m.allocation()
                torch.testing.assert_close(pi.sum(0), torch.ones(3))
                torch.testing.assert_close((pi - .25).sum(0), torch.zeros(3), atol=1e-7, rtol=0)
                explicit = mean + sum(x @ m.basis @ torch.diag(pi[i] - .25) @ m.basis.T
                                      for i, x in enumerate(b))
                torch.testing.assert_close(m.fuse(hidden), explicit)
                self.assertLess(m.orthogonality_loss().item(), 1e-10)

    def test_stage_gradients_and_optimizer_freeze(self):
        for path in PATHS:
            m = load(path).SpatialDsRAProjector(12, 20, 4, 8, 3)
            optimizer = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.1)
            for stage in (1, 2, 3):
                optimizer.zero_grad(set_to_none=True)
                m.set_stage(stage)
                if stage == 2:
                    with torch.no_grad():
                        m.rho.normal_()
                hidden = [torch.randn(2, 6, 12, requires_grad=True) for _ in range(4)]
                before = {name: p.detach().clone() for name, p in m.named_parameters()}
                align, orth = m(hidden, torch.randn(2, 6, 20))
                (align + .01 * orth).backward()
                for h in hidden:
                    self.assertGreater(h.grad.abs().sum().item(), 0)
                self.assertEqual(m.rho.grad is not None, stage == 2)
                self.assertEqual(m.basis.grad is not None, stage == 2)
                optimizer.step()
                for name, p in m.named_parameters():
                    if not p.requires_grad:
                        torch.testing.assert_close(p, before[name], rtol=0, atol=0)

    def test_mask_and_shape_guards(self):
        m = load(PATHS[0]).SpatialDsRAProjector(12, 20, 4, 8, 3, allocation_floor=.05)
        hidden = [torch.randn(2, 6, 12) for _ in range(4)]
        target = torch.randn(2, 6, 20, requires_grad=True)
        mask = torch.tensor([[True, False, True, False, False, False], [False] * 6])
        pred = m.up(m.fuse(hidden))
        loss, _ = m(hidden, target, mask)
        expected = (1 - torch.nn.functional.cosine_similarity(pred, target, dim=-1))[mask].mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(target.grad)
        zero, _ = m(hidden, target, torch.zeros_like(mask))
        self.assertEqual(zero.item(), 0)
        self.assertTrue((m.allocation() >= .05 / 4).all())
        for h, y in [(hidden[:-1], target), (hidden, target[..., :-1]),
                     ([x[..., :-1] for x in hidden], target)]:
            with self.assertRaises(ValueError):
                m(h, y)

    def test_actual_backbone_channel_contracts(self):
        # [Spatial-DsRA v4]: Actual VGGT features have 2048 channels, not embed_dim=1024.
        for path, layers, width in [(PATHS[0], 32, 4096), (PATHS[1], 18, 2048)]:
            m = load(path).SpatialDsRAProjector(width, 2048, layers)
            hidden = [torch.randn(1, 2, width) for _ in range(layers)]
            with torch.no_grad():
                self.assertEqual(m.up(m.fuse(hidden)).shape, (1, 2, 2048))
                loss, orth = m(hidden, torch.randn(1, 2, 2048))
                self.assertTrue(torch.isfinite(loss + orth))

    def test_stage_boundaries_and_state_roundtrip(self):
        module = load(PATHS[0])
        self.assertEqual([module.spatial_stage(s, 10, 20) for s in (0, 9, 10, 19, 20)], [1, 1, 2, 2, 3])
        a = module.SpatialDsRAProjector(12, 20, 4, 8, 3)
        b = module.SpatialDsRAProjector(12, 20, 4, 8, 3)
        with torch.no_grad():
            a.rho.normal_()
        b.load_state_dict(a.state_dict())
        hidden = [torch.randn(1, 2, 12) for _ in range(4)]
        torch.testing.assert_close(a.fuse(hidden), b.fuse(hidden))


if __name__ == '__main__':
    unittest.main()
