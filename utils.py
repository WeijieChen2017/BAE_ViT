import os
import torch
import torch.distributed as dist
# from torch._six import inf
import argparse
from config import get_config
import torch.nn.functional as F
from time import time

def parse_option():
    parser = argparse.ArgumentParser('BAE-ViT training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    # for medical imaging
    parser.add_argument("--max_save", type=int, default=1)
    parser.add_argument("--criterion", type=str, default='l1')
    parser.add_argument("--save_preds", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--is_pretrained", action='store_true')
    parser.add_argument("--pretrained_path", type=str, default='')
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--reg_resume", type=str, default='')
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--multicrop_test", action='store_true')
    parser.add_argument("--multi_model", action='store_true')
    parser.add_argument("--model_ckpts", nargs='+', default=None)
    #parser.add_argument("--freeze_image_branch", type=str, default='')]
    parser.add_argument("--freeze_image_branch", action='store_true')
    parser.add_argument("--gender_dim", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=-1.0)
    parser.add_argument("--gender_filter", type=int, default=-1)
    parser.add_argument("--base_lr", type=float, default=-1.0)
    parser.add_argument("--layer_lr_decay", type=float, default=-1.0)
    parser.add_argument("--gen_heatmap", action='store_true')
    parser.add_argument("--no_test_crop", action='store_true')
    parser.add_argument("--test_pad", action='store_true')
    parser.add_argument("--warmup", type=int, default=-1)
    parser.add_argument("--test_only", action='store_true')

    ## The following argument is abandoned
    parser.add_argument("--task", type=str, default='classification')
    
    

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def get_stats(output, func, logger, return_worst=True):
    # func: either max or min
    # return_worst: If True, then return the filename with the worst val metric; otherwise the ckpt with the best val metric will be returned
    func_name = func.__name__
    assert func_name in ['max', 'min'], f"The function ({func_name}) is not supported for obtaining the statistics of all ckpts"
    reverse = True if func_name == 'max' else False

    checkpoints = os.listdir(output)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    logger.info(f"All checkpoints found in {output}: {checkpoints}")

    num_ckpts = len(checkpoints)
    val_metric_list = []
    for ckpt in checkpoints:
        ckpt_path = os.path.join(output, ckpt)
        ckpt_value = torch.load(ckpt_path, map_location='cpu')
        val_metric = ckpt_value['val_metric']
        val_metric_list.append([ckpt_path, val_metric])

    if len(val_metric_list):
        val_metric_list.sort(reverse=reverse, key=lambda x: x[1])
        path, metric = val_metric_list[-1] if return_worst else val_metric_list[0]
    else:
        path = None
        metric = None
    return path, metric, num_ckpts


def load_checkpoint_weight_only(path, model, logger):
    checkpoint = torch.load(path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    #logger.info(msg)
    del checkpoint
    torch.cuda.empty_cache()

def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        # if 'max_accuracy' in checkpoint:
        #     max_accuracy = checkpoint['max_accuracy']
    if 'opt_val_metric' in checkpoint:
        opt_val_metric = checkpoint['opt_val_metric']
    else:
        opt_val_metric = None

    del checkpoint
    torch.cuda.empty_cache()
    return opt_val_metric

def load_reg_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming form {config.MI.REG_RESUME}....................")
    checkpoint = torch.load(config.MI.REG_RESUME, map_location='cpu')

    model_state = model.state_dict()
    ckpt_state = checkpoint['model']

    ## Drop head weights
    mnames = ['head.weight', 'head.bias']
    if mnames[-1] in model_state:
        if mnames[-1] in ckpt_state and ckpt_state[mnames[-1]].shape != model_state[mnames[-1]].shape:
            for mname in mnames:
                ckpt_state.pop(mname)
    ## Drop attn mask
    for key in model_state:
        if ('attn_mask' in key or 'relative_position_index' in key) and key in ckpt_state:
            ckpt_state.pop(key)
    ## Interpolate relative pos bias table
    all_key_matched = True
    for key in model_state:
        if 'relative_position_bias_table' in key and key in ckpt_state: ## Swin
            # table: (L, num_heads)
            l_target, nh_target = model_state[key].size()
            l_source, nh_source = ckpt_state[key].size()
            if l_target != l_source:
                print(f'Interpolate the position bias table from {l_source} to {l_target}...')
                assert nh_source == nh_target, f"Different number of heads detected! {nh_source} VS {nh_target}..."
                sl_source = int(l_source ** 0.5)
                sl_target = int(l_target ** 0.5)
                value = F.interpolate(ckpt_state[key].permute(1,0).view(1, nh_source, sl_source, sl_source), size=(sl_target, sl_target), mode='bicubic') # (1, nh, l2, l2)

                value = value.reshape(nh_source, l_target).permute(1, 0)
                ckpt_state[key] = value
        elif 'attention_biases' in key and key in ckpt_state:
            nh_target, l_target = model_state[key].size()
            nh_source, l_source = ckpt_state[key].size()
            if l_target != l_source:
                print(f'Interpolate the position bias table from {l_source} to {l_target}...')
                assert nh_source == nh_target, f"Different number of heads detected! {nh_source} VS {nh_target}..."
                sl_source = int(l_source ** 0.5)
                sl_target = int(l_target ** 0.5)
                value = F.interpolate(ckpt_state[key].view(1, nh_source, sl_source, sl_source), size=(sl_target, sl_target), mode='bicubic') # (1, nh, l2, l2)

                value = value.reshape(nh_source, l_target)
                ckpt_state[key] = value
        elif 'local_conv.c' in key and key in ckpt_state:
            _, _, s1, s2 = model_state[key].size()
            _, _, t1, t2 = ckpt_state[key].size()
            if (s1 != t1) or (s2 != t2):
                print(f'Interpolate the local convolution size ({key}) from ({s1},{s2}) to ({t1},{t2})...')
                value = F.interpolate(ckpt_state[key], size=(s1, s2), mode='bicubic')
                ckpt_state[key] = value
        elif key == 'pos_embed': # DeiT
            # pos_embed (1, n, dim)
            _, l_target, dim_target = model_state[key].size()
            _, l_source, dim_source = ckpt_state[key].size()
            if l_target != l_source:
                print(f'Interpolate the position bias table from {l_source} to {l_target}...')
                assert dim_target == dim_source, f"Different number of heads detected! {dim_source} VS {dim_target}..."
                sl_source = int((l_source-1) ** 0.5)
                sl_target = int((l_target-1) ** 0.5)
                ## pos embed for cls token is at the first row
                value = F.interpolate(ckpt_state[key][0, 1:, :].permute(1, 0).view(1, dim_source, sl_source, sl_source), size=(sl_target, sl_target), mode='bicubic')
                value = value.reshape(dim_source, sl_target*sl_target).permute(1, 0)
                value = torch.cat([ckpt_state[key][0, 0, :].unsqueeze(0), value])
                ckpt_state[key] = value.unsqueeze(0)
        elif key == 'cls_token': # DeiT/ViT
            value = ckpt_state[key]
            value = torch.stack([value[0], value[0]])
            ckpt_state[key] = value
        if key in ckpt_state:
            model_state[key] = ckpt_state[key]
        else:
            all_key_matched = False
            print(f"Key not found in the checkpoint: {key}")
    if all_key_matched:
        print(f"All keys successfully matched!")

    msg = model.load_state_dict(model_state, strict=False)
    logger.info(msg)

    del checkpoint
    torch.cuda.empty_cache()


def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, optimizer, lr_scheduler, loss_scaler, val_loss, val_metric, opt_val_metric, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config,
                  'val_loss': val_loss,
                  'val_metric': val_metric,
                  'opt_val_metric': opt_val_metric,
                  }

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    for _ in range(200):
        try:
            torch.save(save_state, save_path)
            logger.info(f"{save_path} saved !!!")
            break
        except:
            logger.info(f"Save file {save_path} failed! re-try after 3 seconds")
            time.sleep(3)


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir, logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    logger.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        #latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=lambda x: int(os.path.basename(x).rsplit('_', 1)[1].split('.')[0]))
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def percentage_loss(y_pred, y_truth, alpha=1.0):
    l1 = ((y_pred - y_truth)/y_truth).abs().mean()
    ld = ((y_pred.mean() - y_truth.mean())/y_truth.mean()).abs()
    return l1 + alpha * ld

def ccc_loss(y_pred, y_truth):
    mean_truth = y_truth.mean()
    mean_pred = y_pred.mean()
    cov = ((y_truth - mean_truth) * (y_pred - mean_pred)).mean()
    var = y_truth.var() + y_pred.var()
    mse = (mean_pred - mean_truth) ** 2
    return (2*cov) / (var + mse)

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

class GatherTensor:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.pred_list = []
        self.target_list = []
        self.all_pred_list = []
        self.all_target_list = []

    def update(self, pred, target):
        self.pred_list.append(pred)
        self.target_list.append(target)

    def sync(self):
        ### Support gathering tensors with different sizes
        n = dist.get_world_size()
        pred_list = torch.cat(self.pred_list)
        target_list = torch.cat(self.target_list)
        local_size = torch.tensor(len(pred_list), device=pred_list.device)
        all_sizes = [torch.tensor(0, device=local_size.device, dtype=local_size.dtype) for _ in range(n)]
        dist.all_gather(all_sizes, local_size)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            new_shape = list(pred_list.shape)
            new_shape[0] = size_diff
            padding = torch.zeros(new_shape, device=pred_list.device, dtype=pred_list.dtype)
            pred_list = torch.cat([pred_list, padding])

            new_shape = list(target_list.shape)
            new_shape[0] = size_diff
            padding = torch.zeros(new_shape, device=target_list.device, dtype=target_list.dtype)
            target_list = torch.cat([target_list, padding])

        all_pred_list = [torch.zeros(pred_list.shape, device=pred_list.device, dtype=pred_list.dtype) for _ in range(n)]
        all_target_list = [torch.zeros(target_list.shape, device=target_list.device, dtype=target_list.dtype) for _ in range(n)]
        dist.all_gather(all_pred_list, pred_list)
        dist.all_gather(all_target_list, target_list)

        for pred, target, size in zip(all_pred_list, all_target_list, all_sizes):
            self.all_pred_list.append(pred[:size])
            self.all_target_list.append(target[:size])