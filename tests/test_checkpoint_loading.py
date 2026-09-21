"""Regression tests for loading checkpoints.

Two separate defects are covered here, both of which silently degrade a run
instead of stopping it:

1. ``remap_checkpoint_keys`` applies the MinkowskiEngine sparse-to-dense GRN
   conversion to every checkpoint, including dense ones. A dense checkpoint
   already stores GRN affines as ``(1, 1, 1, dim)``, so the conversion produced
   a 5-D tensor that matched nothing and left all 36 GRN weights at zero.
2. ``auto_load_model`` reads with torch's default ``weights_only=True``, which
   refuses to unpickle the ``args`` and ``optimizer`` entries that
   ``save_model`` writes into every checkpoint.

Requires timm and tensorboardX, so these run in the training environment rather
than against a bare interpreter.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import utils
from models.self.convnextv2_modified import convnextv2_tiny


class RemapGrnAffineTests(unittest.TestCase):
    def test_dense_grn_affine_keeps_its_shape(self):
        remapped = utils.remap_checkpoint_keys(
            {"stages.0.0.grn.gamma": torch.zeros(1, 1, 1, 384)}
        )

        self.assertEqual(tuple(remapped["stages.0.0.grn.gamma"].shape), (1, 1, 1, 384))

    def test_sparse_grn_affine_is_still_expanded(self):
        remapped = utils.remap_checkpoint_keys(
            {"stages.0.0.grn.gamma": torch.zeros(1, 384)}
        )

        self.assertEqual(tuple(remapped["stages.0.0.grn.gamma"].shape), (1, 1, 1, 384))

    def test_grn_conv_kernel_is_left_alone(self):
        # GRN_SA carries a 7x7 convolution. Its key contains "grn" but it is
        # already dense, so it must not be reshaped.
        remapped = utils.remap_checkpoint_keys(
            {"stages.0.0.grn.conv.weight": torch.zeros(1, 1, 7, 7)}
        )

        self.assertEqual(tuple(remapped["stages.0.0.grn.conv.weight"].shape), (1, 1, 7, 7))


class DenseCheckpointRoundTripTests(unittest.TestCase):
    """The user-visible symptom was one size-mismatch message per GRN affine,
    followed by every GRN weight failing to load."""

    def test_stock_imagenet_checkpoint_matches_the_model_exactly(self):
        model = convnextv2_tiny(num_classes=1000)
        checkpoint = {k: v.clone() for k, v in model.state_dict().items()}

        remapped = utils.remap_checkpoint_keys(checkpoint)

        expected = model.state_dict()
        mismatched = sorted(
            k
            for k, v in remapped.items()
            if k in expected and tuple(v.shape) != tuple(expected[k].shape)
        )
        self.assertEqual(mismatched, [])

    def test_grn_values_actually_reach_the_model(self):
        model = convnextv2_tiny(num_classes=1000)
        gamma = torch.arange(384, dtype=torch.float32).reshape(1, 1, 1, 384)

        remapped = utils.remap_checkpoint_keys({"stages.0.0.grn.gamma": gamma.clone()})
        model.load_state_dict(remapped, strict=False)

        self.assertTrue(torch.equal(model.stages[0][0].grn.gamma, gamma))


class ResumeCheckpointTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = Path(self._tmp.name)

    @staticmethod
    def _args(output_dir, **overrides):
        base = dict(
            output_dir=output_dir,
            auto_resume=False,
            resume="",
            load_encoder_only=False,
            model_ema=False,
            save_ckpt_num=3,
            save_ckpt_freq=1,
            lr=1e-3,
            weight_decay=0.05,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_save_model_output_can_be_read_back(self):
        model = convnextv2_tiny(num_classes=9)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.25)

        utils.save_model(
            args=self._args(self.output_dir),
            epoch=0,
            model=model,
            model_without_ddp=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_scaler=utils.NativeScalerWithGradNormCount(),
        )

        checkpoint_path = self.output_dir / "checkpoint-0.pth"
        self.assertTrue(checkpoint_path.is_file())

        fresh = convnextv2_tiny(num_classes=9)
        utils.auto_load_model(
            args=self._args(self.output_dir, resume=str(checkpoint_path)),
            model=fresh,
            model_without_ddp=fresh,
            optimizer=torch.optim.SGD(fresh.parameters(), lr=0.1),
            loss_scaler=utils.NativeScalerWithGradNormCount(),
        )

        self.assertTrue(torch.equal(fresh.head.weight, model.head.weight))

    def test_checkpoint_carries_the_args_needed_to_configure_eval(self):
        model = convnextv2_tiny(num_classes=9)
        args = self._args(self.output_dir, model="convnextv2_tiny", input_size=200, nb_classes=9)

        utils.save_model(
            args=args,
            epoch=0,
            model=model,
            model_without_ddp=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_scaler=utils.NativeScalerWithGradNormCount(),
        )

        # The eval launcher reads these back so it cannot silently evaluate with
        # a mismatched architecture.
        stored = torch.load(
            self.output_dir / "checkpoint-0.pth", map_location="cpu", weights_only=False
        )["args"]

        self.assertEqual(stored.model, "convnextv2_tiny")
        self.assertEqual(stored.input_size, 200)
        self.assertEqual(stored.nb_classes, 9)


if __name__ == "__main__":
    unittest.main()
