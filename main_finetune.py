# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import datetime
import numpy as np
import time
import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from datasets import build_dataset
from engine_finetune import train_one_epoch, evaluate
from losses import (
    ClassWeightedCrossEntropy,
    ClassWeightedLabelSmoothingCrossEntropy,
    ClassWeightedSoftTargetCrossEntropy,
)

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import str2bool, remap_checkpoint_keys
import models.self.convnextv2_modified as convnextv2

def get_args_parser():
    parser = argparse.ArgumentParser('FCMAE fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='convnextv2_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--drop_path', type=float, default=0., metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--layer_decay_type', type=str, choices=['single', 'group'], default='single',
                        help="""Layer decay strategies. The single strategy assigns a distinct decaying value for each layer,
                        whereas the group strategy assigns the same decaying value for three consecutive layers""")
    
    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')    
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                       help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--copper_loss_weight', type=float, default=1.0,
                        help='Training loss weight for the 铜瘤 class. 1.0 preserves the '
                             'original loss exactly; values > 1 increase the cost of copper '
                             'target mass, including Mixup/CutMix soft targets.')
    
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=0.001, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True,
                        help='Legacy no-op: normalization is always ImageNet stats (pretrained weights require them)')
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    parser.add_argument('--confusion_matrix', type=str2bool, default=True,
                        help='In --eval mode, save confusion matrices (count + row-normalized CSV) and print a summary. The per-epoch validation loop never writes these.')
    parser.add_argument('--save_eval_predictions', type=str2bool, default=True,
                        help='In --eval mode, save per-sample predictions (eval_predictions.csv / eval_misclassified.csv). The per-epoch validation loop never writes these.')
    parser.add_argument('--copper_sweep', type=str2bool, default=False,
                        help='In --eval mode, sweep 铜瘤 logit bias and one-vs-rest probability thresholds.')
    parser.add_argument('--copper_bias_min', type=float, default=0.0)
    parser.add_argument('--copper_bias_max', type=float, default=5.0)
    parser.add_argument('--copper_bias_steps', type=int, default=101)
    parser.add_argument('--copper_threshold_min', type=float, default=0.001)
    parser.add_argument('--copper_threshold_max', type=float, default=0.5)
    parser.add_argument('--copper_threshold_steps', type=int, default=100)
    parser.add_argument('--copper_sweep_min_precision', type=float, default=98.0,
                        help='Precision floor used only to mark/recommend sweep operating points.')
    parser.add_argument('--copper_sweep_max_acc_drop', type=float, default=0.10,
                        help='Maximum Overall Acc@1 drop in percentage points for recommended sweep points.')
    parser.add_argument('--best_acc_tolerance', type=float, default=0.10,
                        help='Checkpoint selection Acc@1 tolerance band in percentage points. '
                             'Within the band from the best Acc@1 seen so far, prefer higher 铜瘤 Recall.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use apex AMP (Automatic Mixed Precision) or not")

    parser.add_argument('--use_grayscale', type=str2bool, default=False,
                        help="Compatibility no-op: the formal pipeline always decodes true grayscale")
    parser.add_argument('--HSVEnhance', type=str2bool, default=False,
                        help='Enable grayscale brightness/contrast jitter during training '
                             '(the grayscale equivalent of the historical HSV-space enhance). '
                             'Note: the pre-fix pipeline applied this unconditionally; it now '
                             'defaults OFF, so bare runs train with a leaner pipeline than before')
    parser.add_argument('--HSVE_factors', type=float, nargs=3, default=[0.05, 0.1, 0.1],
                    help='Legacy no-op: jitter ranges are fixed constants (BRIGHTNESS_RANGE/CONTRAST_RANGE)')
    parser.add_argument('--add_Gauss_Noise', type=str2bool, default=False,
                        help='Add shared-channel Gaussian noise during training '
                             '(off by default; the pre-fix pipeline applied it unconditionally)')
    parser.add_argument('--GN_factors', type=float, nargs=2, default=[0, 25],
                    help='Gauss Noise factors on the 0..255 pixel scale; sigma range = values/255 '
                         '(default (0,25) -> (0, 0.098), ~20x the historical hardcoded (0, 0.005); '
                         'use (0, 1.275) to reproduce the historical strength)')
    parser.add_argument('--load_encoder_only', type=str2bool, default=False,
                        help='Resume only contains weights of encoder or not')
    parser.add_argument('--freeze', type=str2bool, default=False,
                        help='Freeze part of weights when finetuning')
    parser.add_argument('class_labels', type=str, help="Path to class_labels.txt")

    return parser


def _checkpoint_metrics(val_stats):
    return {
        "acc1": float(val_stats["acc1"]),
        "macro_recall": float(val_stats["macro_recall"]),
        "copper_recall": float(val_stats["copper_recall"]),
        "copper_precision": float(val_stats["copper_precision"]),
    }


def _should_update_best_checkpoint(candidate, selected, max_acc_seen, acc_tolerance):
    """Select a business-aware checkpoint without allowing accuracy drift.

    Candidates must stay within acc_tolerance percentage points of the highest
    Acc@1 observed at any epoch. Inside that fixed band, ranking is
    lexicographic: 铜瘤 Recall, 铜瘤 Precision, Macro Recall, then Acc@1.
    """
    floor = max_acc_seen - acc_tolerance
    if candidate["acc1"] < floor - 1e-12:
        return False
    if selected is None or selected["acc1"] < floor - 1e-12:
        return True
    candidate_rank = (
        candidate["copper_recall"],
        candidate["copper_precision"],
        candidate["macro_recall"],
        candidate["acc1"],
    )
    selected_rank = (
        selected["copper_recall"],
        selected["copper_precision"],
        selected["macro_recall"],
        selected["acc1"],
    )
    return candidate_rank > selected_rank


def _write_best_checkpoint_record(output_dir, epoch, metrics, max_acc_seen, acc_tolerance,
                                  filename="best-checkpoint-selection.json"):
    if not output_dir or not utils.is_main_process():
        return
    payload = {
        "epoch": epoch,
        "selection_policy": {
            "acc_tolerance_pp": acc_tolerance,
            "eligibility": "acc1 >= max_acc_seen - acc_tolerance_pp",
            "tie_break_order": [
                "copper_recall",
                "copper_precision",
                "macro_recall",
                "acc1",
            ],
        },
        "max_acc_seen": max_acc_seen,
        "selected_metrics": metrics,
    }
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    try:
        if not args.class_labels.endswith('.txt'):
            raise ValueError(f"Invalid file format: {args.class_labels}. Expected a .txt file.")
        
        # 读取文件内容
        with open(args.class_labels, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            class_list = []
            for line in lines:
                try:
                    _, class_name = line.strip().split(":")
                    class_list.append(class_name)
                except ValueError:
                    raise ValueError(f"Invalid line format: {line.strip()}. Expected 'index:class_name'.")

        if not args.eval:
            if len(class_list) != args.nb_classes:
                raise ValueError(f"Wrong class_label.txt: {args.class_labels} with {len(class_list)} classes! But {args.nb_classes} is right.")
        else:
            args.nb_classes = len(class_list)
    except FileNotFoundError:
        print(f"Error: File '{args.class_labels}' not found.")
        class_list = []
    except ValueError as e:
        print(f"Error: {e}")
        class_list = []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        class_list = []

    if args.copper_loss_weight <= 0:
        raise ValueError(
            f"--copper_loss_weight must be > 0, got {args.copper_loss_weight}"
        )
    if not args.eval and args.copper_loss_weight != 1.0 and "铜瘤" not in class_list:
        raise ValueError(
            "--copper_loss_weight != 1.0 requires a 铜瘤 entry in class_labels"
        )

    if args.eval and class_list and dataset_val is not None:
        dataset_classes = getattr(dataset_val, "classes", None)
        if dataset_classes is not None and len(class_list) != len(dataset_classes):
            raise ValueError(
                f"class_labels.txt has {len(class_list)} classes but the eval dataset "
                f"({getattr(dataset_val, 'root', '?')}) has {len(dataset_classes)}: "
                f"{sorted(dataset_classes)}. Refusing to evaluate on a mismatched class list — "
                f"per-sample predictions would index out of range."
            )

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    model = convnextv2.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        head_init_scale=args.head_init_scale,
    )
    
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        # remove decoder weights
        checkpoint_model_keys = list(checkpoint_model.keys())
        for k in checkpoint_model_keys:
            if 'decoder' in k or 'mask_token'in k or \
               'proj' in k or 'pred' in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        checkpoint_model = remap_checkpoint_keys(checkpoint_model)
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        
        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
        torch.nn.init.constant_(model.head.bias, 0.)

        def freeze(model):
            loaded_keys = set(checkpoint_model.keys())
            
            for name, param in model.named_parameters():
                if (
                    # name.startswith("stages.3") or 
                    name not in loaded_keys or 
                    name.startswith("head")
                ):
                    param.requires_grad = True
                    print(f"Trainable: {name}")
                else:
                    param.requires_grad = False
        
        if args.freeze:
            freeze(model)

        print("\nTrainable Parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    print('number of trainable params:', trainable_parameters)

    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        assert args.layer_decay_type in ['single', 'group']
        if args.layer_decay_type == 'group': # applies for Base and Large models
            num_layers = 12
        else:
            num_layers = sum(model_without_ddp.depths)
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
            depths=model_without_ddp.depths, layer_decay_type=args.layer_decay_type)
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    if args.copper_loss_weight == 1.0:
        # Preserve the historical baseline loss path exactly when cost-sensitive
        # training is disabled.
        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    else:
        copper_idx = class_list.index("铜瘤")
        class_weights = torch.ones(args.nb_classes, dtype=torch.float32)
        class_weights[copper_idx] = args.copper_loss_weight
        if mixup_fn is not None:
            criterion = ClassWeightedSoftTargetCrossEntropy(class_weights)
        elif args.smoothing > 0.:
            criterion = ClassWeightedLabelSmoothingCrossEntropy(
                class_weights, smoothing=args.smoothing
            )
        else:
            criterion = ClassWeightedCrossEntropy(class_weights)
        print(
            f"Cost-sensitive training enabled: 铜瘤 class index={copper_idx}, "
            f"loss weight={args.copper_loss_weight:.4f}"
        )

    criterion = criterion.to(device)
    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        val_stats, acc_info = evaluate(
            data_loader_val, model, device, use_amp=args.use_amp,
            output_dir=args.output_dir,
            save_predictions=args.save_eval_predictions,
            save_confusion=args.confusion_matrix,
            class_names=class_list,
            num_classes=args.nb_classes,
            num_tasks=num_tasks,
            global_rank=global_rank,
            copper_sweep=args.copper_sweep,
            copper_bias_min=args.copper_bias_min,
            copper_bias_max=args.copper_bias_max,
            copper_bias_steps=args.copper_bias_steps,
            copper_threshold_min=args.copper_threshold_min,
            copper_threshold_max=args.copper_threshold_max,
            copper_threshold_steps=args.copper_threshold_steps,
            copper_sweep_min_precision=args.copper_sweep_min_precision,
            copper_sweep_max_acc_drop=args.copper_sweep_max_acc_drop,
        )
        with open(os.path.join(args.output_dir, "eval-class-acc.txt"), "w", encoding="utf-8") as f:
                f.write(f"Overall Acc@1: {val_stats['acc1']:.5f}%\n")
                f.write(f"Macro Recall: {val_stats['macro_recall']:.5f}%\n")
                f.write(f"铜瘤 Recall: {val_stats['copper_recall']:.5f}%\n")
                f.write(f"铜瘤 Precision: {val_stats['copper_precision']:.5f}%\n\n")
                for cls, stats in acc_info.items():
                    num_samples = stats["total"]
                    acc = stats["accuracy"]
                    class_name = class_list[cls]
                    f.write(f"With {num_samples} samples, Class {class_name}: {acc:.2f}%\n")
                f.write("\n")
        print(
            f"Evaluation summary on {len(dataset_val)} images — "
            f"Overall Acc@1: {val_stats['acc1']:.5f}% | "
            f"Macro Recall: {val_stats['macro_recall']:.5f}% | "
            f"铜瘤 Recall: {val_stats['copper_recall']:.5f}% | "
            f"铜瘤 Precision: {val_stats['copper_precision']:.5f}%"
        )
        return
    
    max_accuracy = 0.0
    best_checkpoint_metrics = None
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0
        best_checkpoint_metrics_ema = None

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler, 
            args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        if hasattr(model, 'neck_weights'):
            print("Neck Weights:", convnextv2.LiNormWeight(model.neck_weights).tolist())
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            val_stats, acc_info = evaluate(
                data_loader_val, model, device, use_amp=args.use_amp,
                class_names=class_list, num_classes=args.nb_classes,
                num_tasks=num_tasks, global_rank=global_rank,
            )
            print(
                f"Validation summary on {len(dataset_val)} images — "
                f"Overall Acc@1: {val_stats['acc1']:.5f}% | "
                f"Macro Recall: {val_stats['macro_recall']:.5f}% | "
                f"铜瘤 Recall: {val_stats['copper_recall']:.5f}% | "
                f"铜瘤 Precision: {val_stats['copper_precision']:.5f}%"
            )
            with open(os.path.join(args.output_dir, "val-class-acc.txt"), "a", encoding="utf-8") as f:
                f.write(
                    f"Overall Acc@1: {val_stats['acc1']:.5f}% | "
                    f"Macro Recall: {val_stats['macro_recall']:.5f}% | "
                    f"铜瘤 Recall: {val_stats['copper_recall']:.5f}% | "
                    f"铜瘤 Precision: {val_stats['copper_precision']:.5f}%\n"
                )
                for cls, stats in acc_info.items():
                    num_samples = stats["total"]
                    acc = stats["accuracy"]
                    class_name = class_list[cls]
                    f.write(f"With {num_samples} samples, Class {class_name}: {acc:.2f}%\n")
                f.write("\n")
            candidate_metrics = _checkpoint_metrics(val_stats)
            max_accuracy = max(max_accuracy, candidate_metrics["acc1"])
            if _should_update_best_checkpoint(
                candidate_metrics,
                best_checkpoint_metrics,
                max_accuracy,
                args.best_acc_tolerance,
            ):
                best_checkpoint_metrics = candidate_metrics
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                _write_best_checkpoint_record(
                    args.output_dir,
                    epoch,
                    best_checkpoint_metrics,
                    max_accuracy,
                    args.best_acc_tolerance,
                )
                print(
                    "Updated checkpoint-best.pth at epoch {epoch}: Acc@1 {acc1:.3f}% | "
                    "铜瘤 Recall {copper_recall:.3f}% | 铜瘤 Precision {copper_precision:.3f}% | "
                    "Macro Recall {macro_recall:.3f}%".format(epoch=epoch, **best_checkpoint_metrics)
                )
            print(
                f"Max accuracy seen: {max_accuracy:.2f}% | "
                f"selection floor: {max_accuracy - args.best_acc_tolerance:.2f}%"
            )

            if log_writer is not None:
                log_writer.update(val_acc1=val_stats['acc1'], head="perf", step=epoch)
                log_writer.update(val_acc5=val_stats['acc5'], head="perf", step=epoch)
                log_writer.update(val_loss=val_stats['loss'], head="perf", step=epoch)
                log_writer.update(val_macro_recall=val_stats['macro_recall'], head="perf", step=epoch)
                log_writer.update(val_copper_recall=val_stats['copper_recall'], head="perf", step=epoch)
                log_writer.update(val_copper_precision=val_stats['copper_precision'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         'copper_loss_weight': args.copper_loss_weight,
                         'n_parameters': n_parameters,
                         'trainable_parameters': trainable_parameters}
            
            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                val_stats_ema, _ = evaluate(
                    data_loader_val, model_ema.ema, device, use_amp=args.use_amp,
                    class_names=class_list, num_classes=args.nb_classes,
                    num_tasks=num_tasks, global_rank=global_rank,
                )
                print(f"Accuracy of the model EMA on {len(dataset_val)} val images: {val_stats_ema['acc1']:.1f}%")
                candidate_metrics_ema = _checkpoint_metrics(val_stats_ema)
                max_accuracy_ema = max(max_accuracy_ema, candidate_metrics_ema["acc1"])
                if _should_update_best_checkpoint(
                    candidate_metrics_ema,
                    best_checkpoint_metrics_ema,
                    max_accuracy_ema,
                    args.best_acc_tolerance,
                ):
                    best_checkpoint_metrics_ema = candidate_metrics_ema
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    _write_best_checkpoint_record(
                        args.output_dir,
                        epoch,
                        best_checkpoint_metrics_ema,
                        max_accuracy_ema,
                        args.best_acc_tolerance,
                        filename="best-ema-checkpoint-selection.json",
                    )
                    print(
                        "Updated checkpoint-best-ema.pth at epoch {epoch}: Acc@1 {acc1:.3f}% | "
                        "铜瘤 Recall {copper_recall:.3f}% | 铜瘤 Precision {copper_precision:.3f}% | "
                        "Macro Recall {macro_recall:.3f}%".format(
                            epoch=epoch, **best_checkpoint_metrics_ema
                        )
                    )
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=val_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'val_{k}_ema': v for k, v in val_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'copper_loss_weight': args.copper_loss_weight,
                        'n_parameters': n_parameters,
                         'trainable_parameters': trainable_parameters}
        
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FCMAE fine-tuning', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)