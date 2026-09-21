"""Tests for copper-specific class-weighted training losses."""

import unittest

import torch
import torch.nn.functional as F

from losses import (
    ClassWeightedCrossEntropy,
    ClassWeightedLabelSmoothingCrossEntropy,
    ClassWeightedSoftTargetCrossEntropy,
)


class WeightedLossTests(unittest.TestCase):
    def test_soft_target_unit_weights_match_standard_soft_ce_formula(self):
        logits = torch.tensor(
            [[2.0, 0.5, -0.5], [0.1, 1.2, 0.7]], dtype=torch.float32
        )
        target = torch.tensor(
            [[0.7, 0.3, 0.0], [0.0, 0.4, 0.6]], dtype=torch.float32
        )
        expected = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        actual = ClassWeightedSoftTargetCrossEntropy([1.0, 1.0, 1.0])(
            logits, target
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-7))

    def test_soft_target_copper_weight_scales_fractional_mixup_mass(self):
        logits = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float32)
        # Treat class index 2 as 铜瘤. Half of this Mixup target is copper.
        target = torch.tensor([[0.5, 0.0, 0.5]], dtype=torch.float32)

        base = ClassWeightedSoftTargetCrossEntropy([1.0, 1.0, 1.0])(
            logits, target
        )
        weighted = ClassWeightedSoftTargetCrossEntropy([1.0, 1.0, 2.0])(
            logits, target
        )

        copper_term = -0.5 * F.log_softmax(logits, dim=-1)[0, 2]
        self.assertTrue(torch.allclose(weighted - base, copper_term, atol=1e-7))
        self.assertGreater(weighted.item(), base.item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for device regression test")
    def test_soft_target_accepts_cpu_weights_with_cuda_logits(self):
        # Regression for the remote training crash: criterion/buffer starts on
        # CPU while logits and Mixup soft targets are already on cuda:0.
        criterion = ClassWeightedSoftTargetCrossEntropy([1.0, 1.0, 1.5])
        logits = torch.tensor(
            [[1.2, 0.4, -0.2], [0.1, 0.3, 1.5]],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        target = torch.tensor(
            [[0.7, 0.0, 0.3], [0.0, 0.4, 0.6]],
            dtype=torch.float32,
            device="cuda",
        )
        loss = criterion(logits, target)
        self.assertEqual(loss.device.type, "cuda")
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_hard_label_weight_scales_copper_sample_cost(self):
        logits = torch.tensor([[0.2, 1.5], [1.0, 0.4]], dtype=torch.float32)
        target = torch.tensor([1, 0], dtype=torch.long)
        losses = F.cross_entropy(logits, target, reduction="none")
        expected = (2.0 * losses[0] + losses[1]) / 2.0
        actual = ClassWeightedCrossEntropy([1.0, 2.0])(logits, target)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_label_smoothing_loss_is_finite_and_backpropagates(self):
        logits = torch.tensor(
            [[0.2, 1.5, -0.1], [1.0, 0.4, 0.2]],
            dtype=torch.float32,
            requires_grad=True,
        )
        target = torch.tensor([1, 2], dtype=torch.long)
        criterion = ClassWeightedLabelSmoothingCrossEntropy(
            [1.0, 1.0, 1.5], smoothing=0.1
        )
        loss = criterion(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
