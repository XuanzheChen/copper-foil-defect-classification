import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision import transforms

from datasets import (
    BRIGHTNESS_RANGE,
    CONTRAST_RANGE,
    FORMAL_IMAGE_SIZE,
    GAUSSIAN_NOISE_SIGMA_RANGE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CopperFoilImageFolder,
    GrayscaleToRGB,
    RandomBrightness,
    RandomContrast,
    SharedChannelGaussianNoise,
    build_transform,
    load_grayscale_image,
)


def _args(
    input_size=200,
    hsv_enhance=True,
    add_gauss_noise=False,
    gn_factors=(0.0, 25.0),
):
    """Baseline configuration mirrors train_csd.sh (HSVEnhance on, noise off)."""
    return SimpleNamespace(
        input_size=input_size,
        HSVEnhance=hsv_enhance,
        add_Gauss_Noise=add_gauss_noise,
        GN_factors=list(gn_factors),
    )


def _write_palette_grayscale(path, size=(200, 200)):
    image = Image.new("P", size)
    palette = [0] * (256 * 3)
    palette[0:3] = [37, 37, 37]
    palette[3:6] = [211, 211, 211]
    image.putpalette(palette)

    pixels = [0] * (size[0] * size[1])
    pixels[-1] = 1
    image.putdata(pixels)
    image.save(path)


class PaletteDecodeTests(unittest.TestCase):
    def test_palette_png_is_decoded_through_palette_to_true_grayscale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "palette.png"
            _write_palette_grayscale(path)

            loaded = load_grayscale_image(path)

            self.assertEqual(loaded.mode, "L")
            self.assertEqual(loaded.size, FORMAL_IMAGE_SIZE)
            self.assertEqual(loaded.getpixel((0, 0)), 37)
            self.assertEqual(loaded.getpixel((199, 199)), 211)
            self.assertNotEqual(loaded.getpixel((199, 199)), 1)


class DatasetAdmissionTests(unittest.TestCase):
    def test_non_200x200_sample_is_excluded_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            class_dir = root / "defect"
            class_dir.mkdir(parents=True)

            _write_palette_grayscale(class_dir / "valid.png", (200, 200))
            _write_palette_grayscale(class_dir / "invalid.png", (199, 200))

            report = Path(tmp) / "excluded_non_200x200.csv"
            dataset = CopperFoilImageFolder(
                root=root,
                transform=build_transform(False, _args()),
                exclusion_report_path=report,
            )

            self.assertEqual(len(dataset), 1)
            self.assertEqual(Path(dataset.samples[0][0]).name, "valid.png")
            self.assertEqual(len(dataset.excluded_non_200x200), 1)

            with report.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                list(rows[0].keys()),
                ["path", "class", "width", "height", "reason"],
            )
            self.assertEqual(Path(rows[0]["path"]).name, "invalid.png")
            self.assertEqual(rows[0]["class"], "defect")
            self.assertEqual(rows[0]["width"], "199")
            self.assertEqual(rows[0]["height"], "200")
            self.assertEqual(rows[0]["reason"], "non_200x200")

    def test_loader_defensively_rejects_wrong_original_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong.png"
            Image.new("L", (201, 200), 128).save(path)

            with self.assertRaisesRegex(ValueError, "must be"):
                load_grayscale_image(path)


class TransformPipelineTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("L", FORMAL_IMAGE_SIZE, 128)

    def test_train_baseline_pipeline_has_exact_required_steps_and_ranges(self):
        # Baseline config: HSVEnhance on, Gaussian noise off. Noise is gated by
        # --add_Gauss_Noise and must NOT appear when the flag is false.
        transform = build_transform(True, _args())
        steps = transform.transforms

        self.assertEqual(type(steps[0]).__name__, "Pad")
        self.assertEqual(steps[0].padding, 6)
        self.assertEqual(steps[0].padding_mode, "reflect")

        self.assertEqual(type(steps[1]).__name__, "RandomCrop")
        self.assertEqual(tuple(steps[1].size), FORMAL_IMAGE_SIZE)

        self.assertIsInstance(steps[2], RandomBrightness)
        self.assertEqual(steps[2].factor_range, BRIGHTNESS_RANGE)

        self.assertIsInstance(steps[3], RandomContrast)
        self.assertEqual(steps[3].factor_range, CONTRAST_RANGE)

        self.assertIsInstance(steps[4], GrayscaleToRGB)
        self.assertEqual(type(steps[5]).__name__, "ToTensor")

        self.assertEqual(type(steps[6]).__name__, "Normalize")
        self.assertEqual(tuple(steps[6].mean), IMAGENET_MEAN)
        self.assertEqual(tuple(steps[6].std), IMAGENET_STD)

    def test_gaussian_noise_is_gated_on_flag_and_maps_gn_factors(self):
        no_noise = build_transform(True, _args(add_gauss_noise=False))
        self.assertFalse(
            any(isinstance(step, SharedChannelGaussianNoise) for step in no_noise.transforms)
        )

        with_noise = build_transform(True, _args(add_gauss_noise=True))
        noise_steps = [
            step for step in with_noise.transforms
            if isinstance(step, SharedChannelGaussianNoise)
        ]
        self.assertEqual(len(noise_steps), 1)
        # --GN_factors are on the 0..255 pixel scale; the transform works on
        # [0, 1] tensors, so the upper bound is divided by 255.
        self.assertEqual(noise_steps[0].sigma_range, (0.0, 25.0 / 255.0))

        custom = build_transform(
            True, _args(add_gauss_noise=True, gn_factors=(0.0, 50.0))
        )
        noise_steps = [
            step for step in custom.transforms
            if isinstance(step, SharedChannelGaussianNoise)
        ]
        self.assertEqual(noise_steps[0].sigma_range, (0.0, 50.0 / 255.0))

    def test_color_jitter_is_gated_on_hsv_flag(self):
        with_jitter = build_transform(True, _args(hsv_enhance=True))
        self.assertTrue(
            any(isinstance(step, RandomBrightness) for step in with_jitter.transforms)
        )
        self.assertTrue(
            any(isinstance(step, RandomContrast) for step in with_jitter.transforms)
        )

        without_jitter = build_transform(True, _args(hsv_enhance=False))
        self.assertFalse(
            any(isinstance(step, RandomBrightness) for step in without_jitter.transforms)
        )
        self.assertFalse(
            any(isinstance(step, RandomContrast) for step in without_jitter.transforms)
        )

    def test_noise_lands_after_tensor_before_normalize(self):
        transform = build_transform(True, _args(add_gauss_noise=True))
        steps = transform.transforms

        self.assertEqual(type(steps[5]).__name__, "ToTensor")
        self.assertIsInstance(steps[6], SharedChannelGaussianNoise)
        self.assertEqual(type(steps[7]).__name__, "Normalize")

    def test_train_tensor_shape_channels_and_values_are_valid(self):
        train_transform = build_transform(True, _args())
        pre_normalize = transforms.Compose(train_transform.transforms[:-1])

        torch.manual_seed(7)
        x = pre_normalize(self.image)

        self.assertEqual(tuple(x.shape), (3, 200, 200))
        self.assertTrue(torch.equal(x[0], x[1]))
        self.assertTrue(torch.equal(x[1], x[2]))
        self.assertGreaterEqual(float(x.min()), 0.0)
        self.assertLessEqual(float(x.max()), 1.0)
        self.assertTrue(torch.isfinite(x).all())

        torch.manual_seed(7)
        normalized = train_transform(self.image)
        self.assertEqual(tuple(normalized.shape), (3, 200, 200))
        self.assertTrue(torch.isfinite(normalized).all())

    def test_shared_gaussian_noise_preserves_identical_channels_and_clamps(self):
        noise = SharedChannelGaussianNoise((0.005, 0.005))
        x = torch.ones(3, 200, 200)

        torch.manual_seed(13)
        y = noise(x)

        self.assertEqual(tuple(y.shape), (3, 200, 200))
        self.assertTrue(torch.equal(y[0], y[1]))
        self.assertTrue(torch.equal(y[1], y[2]))
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)
        self.assertTrue(torch.isfinite(y).all())

    def test_eval_is_deterministic_and_has_expected_shape(self):
        eval_transform = build_transform(False, _args())

        first = eval_transform(self.image)
        second = eval_transform(self.image)

        self.assertEqual(tuple(first.shape), (3, 200, 200))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())

        pre_normalize = transforms.Compose(eval_transform.transforms[:-1])
        rgb_tensor = pre_normalize(self.image)
        self.assertTrue(torch.equal(rgb_tensor[0], rgb_tensor[1]))
        self.assertTrue(torch.equal(rgb_tensor[1], rgb_tensor[2]))

    def test_formal_pipelines_contain_no_resize_or_other_forbidden_augmentations(self):
        forbidden_names = {
            "Resize",
            "RandomResize",
            "RandomResizedCrop",
            "Scale",
            "RandomHorizontalFlip",
            "RandomVerticalFlip",
            "RandomRotation",
            "RandomErasing",
            "RandomPerspective",
            "RandomAffine",
            "AutoAugment",
            "RandAugment",
            "TrivialAugmentWide",
        }

        configs = [
            _args(),
            _args(hsv_enhance=False),
            _args(add_gauss_noise=True),
            _args(hsv_enhance=False, add_gauss_noise=True),
        ]
        for is_train in (True, False):
            for config in configs:
                pipeline = build_transform(is_train, config)
                actual_names = {type(step).__name__ for step in pipeline.transforms}
                self.assertTrue(
                    forbidden_names.isdisjoint(actual_names),
                    msg=f"Forbidden transform found: {forbidden_names & actual_names}",
                )

    def test_input_size_224_is_rejected_instead_of_resized(self):
        with self.assertRaisesRegex(ValueError, "requires --input_size 200"):
            build_transform(True, _args(input_size=224))


if __name__ == "__main__":
    unittest.main()
