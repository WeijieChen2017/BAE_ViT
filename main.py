import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, load_checkpoint_weight_only, load_reg_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor, parse_option, get_stats, GatherTensor, \
    percentage_loss

from data.build import med_multi_crop_transform


def main(config):
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn = build_loader(config)
    if config.DATA.DATASET == 'rsna':
        ## This is a regression dataset
        if config.MI.CRITERION == 'l1':
            criterion = torch.nn.L1Loss()
        elif config.MI.CRITERION == 'mse':
            criterion = torch.nn.MSELoss()
        elif config.MI.CRITERION == 'percentage':
            criterion = lambda x, y: percentage_loss(x, y, config.MI.ALPHA)
        else:
            raise ValueError(f"The criterion {config.MI.CRITERION} is not supported for RSNA data")

        if config.MI.MULTI_CROP:
            validate = validate_rsna_multi_crop
        else:
            validate = validate_rsna

        train_one_epoch = train_one_epoch_rsna
        compare = min
        opt_val_metric = 1.0e5
    else:
        ## The default task is classification
        raise ValueError(f"dataset: {config.DATA.DATASET} is not supported!")

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    if hasattr(model, 'load_pretrained_model') and config.MI.IS_PRETRAINED:
        model.load_pretrained_model(config.MI.PRETRAINED_PATH)

    model.cuda()
    model_without_ddp = model

    if config.MI.FREEZE_IMG_BRANCH:
        model_without_ddp.freeze_image_branch()

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.MI.MULTI_MODEL and config.EVAL_MODE:
        val_result = validate(config, model, criterion, data_loader_val, logger, model_without_ddp=model_without_ddp)
        val_loss = val_result['loss']
        logger.info(f"Metric of the network on the {len(dataset_val)} validation images: {val_loss:.5f}")

        if data_loader_test is not None:
            test_result = validate(config, model, criterion, data_loader_test, logger, prefix='Test', model_without_ddp=model_without_ddp)
            num_test_data = test_result['num_test_data']
            test_loss = test_result['loss']
            logger.info(f"Loss of the network on the {len(dataset_test)}({num_test_data} volumes) test images: {test_loss:.5f}")
        return


    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        opt_val_metric = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        if config.MI.GEN_HEATMAP:
            model = model_without_ddp
        
        if not config.MI.TEST_ONLY:
            val_result = validate(config, model, criterion, data_loader_val, logger)
            val_loss = val_result['loss']
            logger.info(f"Metric of the network on the {len(dataset_val)} validation images: {val_loss:.5f}")

        if data_loader_test is not None:
            test_result = validate(config, model, criterion, data_loader_test, logger, prefix='Test')
            num_test_data = test_result['num_test_data']
            test_loss = test_result['loss']
            logger.info(f"Loss of the network on the {len(dataset_test)}({num_test_data} volumes) test images: {test_loss:.5f}")
        
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME) and (not config.MI.REG_RESUME):
        load_pretrained(config, model_without_ddp, logger)
        # acc1, acc5, loss = validate(config, data_loader_val, model)
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        val_result = validate(config, model, criterion, data_loader_val, logger)
        val_loss = val_result['loss']
        logger.info(f"Metric of the network on the {len(dataset_val)} validation images: {val_loss:.5f}")

        if data_loader_test is not None:
            test_result = validate(config, model, criterion, data_loader_test, logger, prefix='Test')
            num_test_data = test_result['num_test_data']
            test_loss = test_result['loss']
            logger.info(f"Loss of the network on the {len(dataset_test)}({num_test_data} volumes) test images: {test_loss:.5f}")

    if config.MI.REG_RESUME and (not config.MODEL.RESUME):
        load_reg_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        val_result = validate(config, model, criterion, data_loader_val, logger)
        val_loss = val_result['loss']
        logger.info(f"Metric of the network on the {len(dataset_val)} validation images: {val_loss:.5f}")

        if data_loader_test is not None:
            test_result = validate(config, model, criterion, data_loader_test, logger, prefix='Test')
            num_test_data = test_result['num_test_data']
            test_loss = test_result['loss']
            logger.info(f"Loss of the network on the {len(dataset_test)}({num_test_data} volumes) test images: {test_loss:.5f}")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, lr_scheduler, epoch, mixup_fn, 
                        loss_scaler, logger)

        val_result = validate(config, model, criterion, data_loader_val, logger)
        val_loss = val_result['loss']
        val_metric = val_result['metric']
        opt_val_metric = compare(opt_val_metric, val_metric)

        if dist.get_rank() == 0:
            file_path_worst, _, num_ckpts = get_stats(output=config.OUTPUT, func=compare, logger=logger)
            logger.info(f"Saving ckpt_epoch_{epoch}.pth...")
            if num_ckpts > config.MI.MAX_SAVE:
                os.remove(file_path_worst)
            save_checkpoint(config, epoch, model_without_ddp, optimizer, lr_scheduler, loss_scaler, val_loss, val_metric, opt_val_metric,
                          logger)

        logger.info(f"Metric of the network on the {len(dataset_val)} validation images: {val_metric:.5f}")
        logger.info(f'Optimal metric: {opt_val_metric:.5f}')

        if data_loader_test is not None:
            test_result = validate(config, model, criterion, data_loader_test, logger, prefix='Test')
            num_test_data = test_result['num_test_data']
            test_loss = test_result['loss']
            logger.info(f"Loss of the network on the {len(dataset_test)}({num_test_data} volumes) test images: {test_loss:.5f}")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch_rsna(config, model, criterion, data_loader, optimizer, lr_scheduler, epoch, mixup_fn, loss_scaler, logger):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if config.MI.IS_GENDER_INFO:
            gender = targets[:,:2].detach().clone().cuda(non_blocking=True)
            samples = (samples, gender)
        targets = targets[:, 2]

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples).squeeze(-1)

        assert outputs.shape == targets.shape, f"output shape {outputs.shape} VS target shape {targets.shape}"
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)

        end = time.time()

        if idx % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f} \t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.5f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}\t"
                f"train-loss {loss_meter.avg:.5f}")


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

@torch.no_grad()
def validate_rsna(config, model, criterion, data_loader, logger, prefix='Val', **kwargs):

    model.eval()

    label_mean = 118.9484375
    label_std = 50.01946962242396

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()
    end = time.time()

    criterion_MAE = torch.nn.L1Loss()

    if config.MI.SAVE_PREDS:
        pred_list = []
        truth_list = []

    if config.MI.DEBUG:
        criterion2 = torch.nn.L1Loss(reduction='none')
        all_metrics = []

    N = len(data_loader)
    logger.info(f"Predicting...")
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        #targets = targets.unsqueeze(1).cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        images = samples
        gender = None

        if config.MI.IS_GENDER_INFO:
            gender = targets[:,:2].detach().clone().cuda(non_blocking=True)
            samples = (samples, gender)
        targets = targets[:, 2]

        if config.MI.GEN_HEATMAP and prefix=='Test':
        #if config.MI.GEN_HEATMAP:
            from gen_plots import gen_attn_heatmap
            #cam_name = 'GradCAM'
            cam_name = 'ScoreCAM'
            outputs = gen_attn_heatmap(config.MODEL.NAME, model, images, gender, save_name=os.path.join(config.OUTPUT, f"{cam_name}_{prefix}_{idx}.png"), image_mean=config.MI.IMAGE_MEAN, image_std=config.MI.IMAGE_STD, gts=targets, cam_name=cam_name)
        else:
            outputs = model(samples).squeeze(-1)

        assert outputs.shape == targets.shape, f"output shape {outputs.shape} VS target shape {targets.shape}"
        loss = criterion(outputs, targets)
        metric = criterion_MAE((outputs*label_std+label_mean), (targets*label_std+label_mean))

        if config.MI.DEBUG:
            all_metrics.append(criterion2((outputs*label_std+label_mean), (targets*label_std+label_mean)).detach().cpu().numpy())

        loss = reduce_tensor(loss)
        metric = reduce_tensor(metric)

        loss_meter.update(loss.item(), targets.size(0))
        metric_meter.update(metric.item(), targets.size(0))
        batch_time.update(time.time() - end)

        if config.MI.SAVE_PREDS:
            pred_list.append((outputs*label_std+label_mean).detach().cpu().numpy())
            truth_list.append((targets*label_std+label_mean).detach().cpu().numpy())

        end = time.time()

        if idx % 10 == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            info = (
                f'{prefix}: [{idx}/{N}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.5f} ({loss_meter.avg:.5f})\t'
                f'Metric {metric_meter.val:.5f} ({metric_meter.avg:.5f})\t'
                f'Mem {memory_used:.0f}MB'
                )
            logger.info(info)

    logger.info(f"*{prefix}- Final loss: {loss_meter.avg:.5f}; Final metric: {metric_meter.avg:.5f}"
        )

    if config.MI.SAVE_PREDS:
        logger.info(f"Saving predictions...")
        np.save(os.path.join(config.OUTPUT, 'predictions.npy'), np.concatenate(pred_list))
        logger.info(f'MAE: {np.abs(np.concatenate(pred_list)-np.concatenate(truth_list)).mean()}')

    if config.MI.DEBUG:
        logger.info(f"Saving all metrics...")
        np.save(os.path.join(config.OUTPUT, 'all_metrics.npy'), np.concatenate(all_metrics))

    result = dict(loss=loss_meter.avg, metric=metric_meter.avg, num_test_data=N)
    return result

@torch.no_grad()
def validate_rsna_multi_crop(config, model, criterion, data_loader, logger, prefix='Val', **kwargs):
    # collection_list=[97, 71, 56, 73, 15,79, 69,53,94,62, 21, 95, 24, 95, 28, 25, 31]
    model.eval()

    multi_crop_transform = med_multi_crop_transform(
                            img_size = config.DATA.IMG_SIZE,
                            interpolation = config.DATA.INTERPOLATION, 
                            mean = config.MI.IMAGE_MEAN, 
                            std = config.MI.IMAGE_STD,
                            is_no_crop = config.MI.GEN_HEATMAP,
        )

    label_mean = 118.9484375
    label_std = 50.01946962242396

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()
    end = time.time()

    criterion_MAE = torch.nn.L1Loss()

    if config.MI.SAVE_PREDS:
        pred_list = []
        truth_list = []

    if config.MI.DEBUG:
        criterion2 = torch.nn.L1Loss(reduction='none')
        all_metrics = []

    N = len(data_loader)
    logger.info(f"Predicting...")
    for idx, (samples, targets) in enumerate(data_loader):
        targets = targets.cuda(non_blocking=True)

        outputs = []
        for i, sample in enumerate(samples):
            crop_list = []
            for _ in range(config.MI.MULTI_CROP_NUM):
                crop_list.append(multi_crop_transform(sample).unsqueeze(0))
            crops = torch.cat(crop_list).cuda()
            if config.MI.IS_GENDER_INFO:
                gender = targets[i,:2].detach().clone().cuda(non_blocking=True)
                gender = gender.repeat(config.MI.MULTI_CROP_NUM, 1)
                crops = (crops, gender)
            outputs.append(model(crops).median().item())
        outputs = torch.tensor(outputs).cuda()
        targets = targets[:, 2]

        assert outputs.shape == targets.shape, f"output shape {outputs.shape} VS target shape {targets.shape}"
        loss = criterion(outputs, targets)
        metric = criterion_MAE((outputs*label_std+label_mean), (targets*label_std+label_mean))

        if config.MI.DEBUG:
            all_metrics.append(criterion2((outputs*label_std+label_mean), (targets*label_std+label_mean)).detach().cpu().numpy())

        loss = reduce_tensor(loss)
        metric = reduce_tensor(metric)

        loss_meter.update(loss.item(), targets.size(0))
        metric_meter.update(metric.item(), targets.size(0))
        batch_time.update(time.time() - end)

        if config.MI.SAVE_PREDS:
            pred_list.append((outputs*label_std+label_mean).detach().cpu().numpy())
            truth_list.append((targets*label_std+label_mean).detach().cpu().numpy())

        end = time.time()

        if idx % 10 == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            info = (
                f'{prefix}: [{idx}/{N}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.5f} ({loss_meter.avg:.5f})\t'
                f'Metric {metric_meter.val:.5f} ({metric_meter.avg:.5f})\t'
                f'Mem {memory_used:.0f}MB'
                )
            logger.info(info)

    logger.info(f"*{prefix}- Final loss: {loss_meter.avg:.5f}; Final metric: {metric_meter.avg:.5f}"
        )

    if config.MI.SAVE_PREDS:
        logger.info(f"Saving predictions...")
        np.save(os.path.join(config.OUTPUT, 'predictions.npy'), np.concatenate(pred_list))
        logger.info(f'MAE: {np.abs(np.concatenate(pred_list)-np.concatenate(truth_list)).mean()}')

    if config.MI.DEBUG:
        logger.info(f"Saving all metrics...")
        np.save(os.path.join(config.OUTPUT, 'all_metrics.npy'), np.concatenate(all_metrics))

    result = dict(loss=loss_meter.avg, metric=metric_meter.avg, num_test_data=N)
    return result


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed+6)
    torch.cuda.manual_seed(seed+6)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
