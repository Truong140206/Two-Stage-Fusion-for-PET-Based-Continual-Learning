"""Train the LoRA pool or evaluate saved checkpoints with RP fusion."""

import datetime
import os
import time

import numpy as np
import torch
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import utils
from datasets import build_continual_dataloader
from engines.hrm_lora_wtp_and_tap_engine import (
    calibrate_rp_head,
    compute_rp_statistics,
    evaluate_till_now,
    pin_rp_extractor,
    train_and_evaluate,
)
from engines.random_projection_head import begin_rp_task, reset_rp_head, solve_rp_head
import vits.hrm_lora_vision_transformer as hide_lora_vision_transformer


def train(args):
    device = torch.device(args.device)
    data_loader, data_loader_per_cls, class_mask, target_task_map = (
        build_continual_dataloader(args)
    )
    print(f"Creating original model: {args.original_model}")
    original_model = create_model(
        args.original_model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        mlp_structure=args.original_model_mlp_structure,
    )
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        lora=True,
        lora_type=args.lora_type,
        rank=args.lora_rank,
        lora_pool_size=args.size,
    )
    original_model.to(device)
    model.to(device)
    for _, parameter in original_model.named_parameters():
        parameter.requires_grad = False
    if args.freeze:
        for name, parameter in model.named_parameters():
            if name.startswith(tuple(args.freeze)):
                parameter.requires_grad = False

    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        rp_head_enabled = bool(getattr(args, 'rp_head', False))
        if rp_head_enabled:
            reset_rp_head()
            print(
                'RP head: dim=', getattr(args, 'rp_dim', 5000),
                'activation=', getattr(args, 'rp_activation', 'relu'),
                'lambda=', getattr(args, 'rp_lambda', 1e4),
                'source=', getattr(args, 'rp_feature_source', 'original'),
            )

        task_count = args.num_tasks
        if args.max_train_tasks > 0:
            task_count = min(task_count, args.max_train_tasks)
        for task_id in range(task_count):
            checkpoint_path = os.path.join(
                args.output_dir, f'checkpoint/task{task_id + 1}_checkpoint.pth'
            )
            if not os.path.exists(checkpoint_path):
                print('No checkpoint found at:', checkpoint_path)
                return
            print('Loading checkpoint from:', checkpoint_path)
            checkpoint = utils.load_checkpoint(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])

            original_checkpoint_path = os.path.join(
                args.trained_original_model,
                f'checkpoint/task{task_id + 1}_checkpoint.pth',
            )
            if not os.path.exists(original_checkpoint_path):
                print('No checkpoint found at:', original_checkpoint_path)
                return
            print('Loading checkpoint from:', original_checkpoint_path)
            original_checkpoint = utils.load_checkpoint(
                original_checkpoint_path, map_location=device
            )
            original_model.load_state_dict(original_checkpoint['model'])

            if rp_head_enabled:
                pin_rp_extractor(original_model, args)
                begin_rp_task(args)
                print('Accumulating RP-head statistics for task', task_id + 1)
                compute_rp_statistics(
                    model=model,
                    original_model=original_model,
                    data_loader=data_loader_per_cls,
                    device=device,
                    task_id=task_id,
                    class_mask=class_mask[task_id],
                    args=args,
                )
                seen_class_ids = [
                    int(class_id)
                    for task in range(task_id + 1)
                    for class_id in class_mask[task]
                ]
                solve_rp_head(args, device, seen_class_ids=seen_class_ids)
                print('RP head solved over', len(seen_class_ids), 'seen classes')
                if bool(getattr(args, 'rp_calibrate', False)):
                    temperature = calibrate_rp_head(
                        model=model,
                        original_model=original_model,
                        data_loader=data_loader_per_cls,
                        device=device,
                        task_id=task_id,
                        class_mask=class_mask[task_id],
                        args=args,
                    )
                    print('RP head temperature:', temperature)

            evaluate_till_now(
                model, original_model, data_loader, device,
                task_id, class_mask, target_task_map, acc_matrix, args,
            )
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    n_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print('number of params:', n_parameters)

    global_batch_size = (
        args.batch_size
        if args.unscale_lr
        else args.batch_size * args.world_size
    )
    args.lr = args.lr * global_batch_size / 256.0
    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = (
        create_scheduler(args, optimizer)[0]
        if args.sched != 'constant'
        else None
    )
    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_and_evaluate(
        model, model_without_ddp, original_model,
        criterion, data_loader, data_loader_per_cls,
        optimizer, lr_scheduler, device, class_mask, target_task_map, args,
    )
    total_time = time.time() - start_time
    print(f"Total training time: {datetime.timedelta(seconds=int(total_time))}")
