# Copper Foil Defect Classification

This repository contains the research code selected for the new copper-foil defect image classification project. It is based on Meta's ConvNeXt V2 implementation and includes the custom model variants developed in the earlier project.

## Recommended route

1. Build a baseline with `convnextv2_tiny` and ImageNet pretrained weights.
2. Evaluate `convnextv2_grn_sa_tiny` and `convnextv2_fusion_tiny` separately.
3. Use `convnextv2_fusion_grn_sa_tiny` as the combined ConvNeXt V2-CSD candidate.
4. Compare overall accuracy, per-class accuracy, recall and inference FPS on the new dataset.

The new project does not assume label re-clustering, long-tail resampling or dedicated few-shot augmentation. These techniques should only be introduced if error analysis shows a clear need.

## Dataset layout

The default `IMNET` loader expects an ImageFolder-style directory:

```text
dataset_root/
  train/
    class_a/
    class_b/
  val/
    class_a/
    class_b/
```

The class label file is a required positional argument. Each line must use the format `index:class_name`, for example:

```text
0:class_a
1:class_b
```

## Environment

The legacy reference environment is Python 3.8, PyTorch 1.9.0, TorchVision 0.10.0 and CUDA 11.1. Install the Python packages with:

```bash
pip install -r requirements.txt
```

MinkowskiEngine is only needed for FCMAE self-supervised pretraining. Standard supervised fine-tuning does not use the pretraining entry point.

## Training

Edit the paths in `train_csd.sh`, then run it on a Linux training machine. The script uses batch size 128, 200 epochs, 30 warmup epochs, 96 x 96 input, Mixup 0.8 and CutMix 1.0. With one GPU and `update_freq=1`, the configured base learning rate produces an actual learning rate of 0.0016.

Pretrained weights and image data are intentionally excluded from Git. Store them in external data or artifact storage.

## Important implementation notes

- The dense classification entry point is `main_finetune.py`.
- Custom models are defined in `models/self/convnextv2_modified.py`.
- `convnextv2_fusion_grn_sa_tiny` is the combined CSD model factory.
- The current code is a recovered research snapshot. Before production use, validate transforms, class mapping and metrics on the new dataset.

## License

The upstream ConvNeXt V2 code retains its original license. See `LICENSE` for details.

